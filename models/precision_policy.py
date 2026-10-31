"""Precision-only adapters. No weights or mathematical operators are replaced."""
from __future__ import annotations
from dataclasses import asdict,dataclass
import torch
import math

@dataclass(frozen=True)
class PrecisionPolicy:
    outer_autocast: str = "bf16"
    composer: str = "bf16"
    matmul_precision: str = "ieee"
    score_dtype: str = "existing"
    reader: str = "bf16"
    scaler_init_scale: float = 1024.0
    scaler_growth_interval: int = 2000

    def __post_init__(self):
        if self.outer_autocast not in ("bf16","fp32"): raise ValueError("outer_autocast must be bf16/fp32")
        if self.composer not in ("bf16","fp32_ieee","fp32_tf32","fp16"): raise ValueError("Unknown Composer precision")
        if self.matmul_precision not in ("ieee","tf32"): raise ValueError("Unknown matmul precision")
        if self.reader not in ("bf16","fp32_ieee"): raise ValueError("Unknown Reader precision")
        if self.reader=="fp32_ieee" and self.matmul_precision!="ieee": raise ValueError("FP32 Reader requires IEEE matmul")
        if self.score_dtype not in ("existing","fp32"): raise ValueError("Unknown score dtype")
        if not math.isfinite(self.scaler_init_scale) or self.scaler_init_scale<=0 or self.scaler_growth_interval<1:
            raise ValueError("Invalid GradScaler settings")
        expected="tf32" if self.composer=="fp32_tf32" else "ieee"
        if self.matmul_precision!=expected: raise ValueError("Composer mode and process matmul precision conflict")

    def state_dict(self): return asdict(self)

    def configure_backend(self):
        # Once at worker startup. Never mix old allow_tf32 setters with this API.
        if not hasattr(torch.backends.cuda.matmul,"fp32_precision"):
            raise RuntimeError("This study requires PyTorch's fp32_precision API")
        torch.backends.fp32_precision="ieee"
        torch.backends.cuda.matmul.fp32_precision=self.matmul_precision
        if torch.backends.cuda.matmul.fp32_precision!=self.matmul_precision: raise RuntimeError("Backend policy not applied")
        return backend_state()

    def outer_context(self,device):
        device=torch.device(device)
        return torch.autocast(device.type,dtype=torch.bfloat16,
                              enabled=device.type=="cuda" and self.outer_autocast=="bf16")

    def compose(self,module,left,right):
        device=left.device.type
        if self.composer.startswith("fp32"):
            if device=="cuda" and torch.backends.cuda.matmul.fp32_precision!=self.matmul_precision:
                raise RuntimeError("FP32 Composer backend changed after process initialization")
            with torch.autocast(device,enabled=False):
                result=module(left.float(),right.float())
        else:
            dtype=torch.float16 if self.composer=="fp16" else torch.bfloat16
            with torch.autocast(device,dtype=dtype,enabled=device=="cuda"):
                if self.composer=="fp16" and device=="cuda": left,right=left.to(dtype),right.to(dtype)
                result=module(left,right)
            # AMP LayerNorm emits FP32 on this build. FP16 storage is explicit.
            if self.composer=="fp16" and device=="cuda": result=result.to(torch.float16)
        observer=getattr(module,"precision_state_trace",None)
        if observer is not None: observer(module,result)
        return result

    def make_scaler(self,device="cuda"):
        return torch.amp.GradScaler(torch.device(device).type,
            enabled=torch.device(device).type=="cuda" and self.composer=="fp16",
            init_scale=self.scaler_init_scale,growth_interval=self.scaler_growth_interval)


def backend_state():
    return dict(generic_fp32=torch.backends.fp32_precision,matmul_fp32=torch.backends.cuda.matmul.fp32_precision,
        cudnn_fp32=torch.backends.cudnn.fp32_precision,
        fp16_reduced_reduction=torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
        bf16_reduced_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        fp16_accumulation=getattr(torch.backends.cuda.matmul,"allow_fp16_accumulation",None))


def configure_model(model,policy):
    if isinstance(policy,dict): policy=PrecisionPolicy(**policy)
    if any(p.dtype!=torch.float32 for p in model.parameters() if p.is_floating_point()):
        raise ValueError("Keep all master parameters FP32")
    if getattr(model,"precision_policy",None) is not None:
        if model.precision_policy!=policy: raise ValueError("Create a fresh model to change precision policy")
        return model
    model.precision_policy=policy
    for reader in model.readers.values(): reader.precision_mode=policy.reader
    if model.forest is not None:
        model.forest.precision_policy=policy
        if policy.score_dtype=="fp32":
            def cast_scores(module,args):
                leaves,scores,coeff,mask,*rest=args
                return (leaves,scores.float(),coeff,mask,*rest)
            model.forest.register_forward_pre_hook(cast_scores)
    return model


POLICIES={
    "A":PrecisionPolicy(),
    "B":PrecisionPolicy(composer="fp32_ieee"),
    "C":PrecisionPolicy(composer="fp32_tf32",matmul_precision="tf32"),
    "D":PrecisionPolicy(composer="fp16"),
    "E":PrecisionPolicy(outer_autocast="fp32",composer="fp32_ieee",reader="fp32_ieee"),
    "S":PrecisionPolicy(composer="fp32_ieee",score_dtype="fp32"),
}
