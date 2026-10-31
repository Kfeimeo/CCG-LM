"""Read-only native SDPA probes and shared Reader-study policies."""
from __future__ import annotations
from dataclasses import replace
from contextlib import nullcontext
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from models.precision_policy import PrecisionPolicy

READER_POLICIES={
    "B":PrecisionPolicy(composer="fp32_ieee",reader="bf16"),
    "R":PrecisionPolicy(composer="fp32_ieee",reader="fp32_ieee"),
    "E":PrecisionPolicy(outer_autocast="fp32",composer="fp32_ieee",reader="fp32_ieee"),
}

class AttentionProbe(TorchDispatchMode):
    """Observe actual MATH operators without replacing SDPA or drawing RNG."""
    def __init__(self,observer):super().__init__();self.observer=observer;self.bmms=0
    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        out=func(*args,**(kwargs or {}));name=str(func)
        self.observer.operations.append(name)
        if name=="aten._safe_softmax.default":
            self.observer.record("attention_logits",args[0]);self.observer.record("attention_probabilities",out)
        elif name=="aten.native_dropout.default":self.observer.record("attention_dropped_probabilities",out[0])
        elif name=="aten.bmm.default":
            self.bmms+=1
            if self.bmms==2:self.observer.record("attention_aggregation_internal",out)
        return out

class ReaderObserver:
    def __init__(self,save_tensors=True,capture_attention=True):
        self.values={};self.specs={};self.operations=[];self.backend={}
        self.save_tensors,self.capture_attention=save_tensors,capture_attention
    def record(self,name,value):
        self.specs[name]=dict(actual_dtype=str(value.dtype),tensor_shape=list(value.shape))
        if self.save_tensors:self.values[name]=value.detach().cpu().clone()
    def sdpa(self,q,k,v,bias,path):
        self.backend=dict(policy=path,query_dtype=str(q.dtype),key_dtype=str(k.dtype),value_dtype=str(v.dtype),
                          bias_dtype=str(bias.dtype),dropout_preserved=True)
    def attention_context(self):return AttentionProbe(self) if self.capture_attention else nullcontext()
    def summary(self):return dict(tensors=self.specs,sdpa=self.backend,operations=self.operations)


def tensor_errors(a,b):
    # Masked attention logits legitimately contain matching -inf entries.
    x,y=a.detach().float().cpu(),b.detach().float().cpu()
    assert x.shape==y.shape
    finite=torch.isfinite(x)&torch.isfinite(y)
    bad=(~finite)&(~((x==y)&~torch.isnan(x)&~torch.isnan(y)))
    xx,yy=x[finite],y[finite];d=xx-yy
    norm=float(xx.double().norm()) if xx.numel() else 0.
    return dict(max_absolute_error=float(d.abs().max()) if d.numel() else 0.,
        normalized_l2_error=float(d.double().norm())/max(norm,1e-30) if d.numel() else 0.,
        rms_error=float(d.double().square().mean().sqrt()) if d.numel() else 0.,
        reference_norm=norm,actual_norm=float(yy.double().norm()) if yy.numel() else 0.,
        actual_dtype=[str(a.dtype),str(b.dtype)],tensor_shape=list(a.shape),
        compared_finite_elements=xx.numel(),matching_masked_infinities=int(((x==y)&torch.isinf(x)).sum()),
        nonfinite_mismatches=int(bad.sum()),nan_reference=int(torch.isnan(x).sum()),nan_actual=int(torch.isnan(y).sum()))


def global_gradients(a,b,strict_label="bf16"):
    from precision_study import gradient_report
    from validate_cky_v25 import gradient_statistics
    modules=gradient_report(a,b);all_stats=gradient_statistics(a,b)
    failures=sorted({n for m in modules.values() for n in m.get("strict_"+strict_label+"_failures",[])})
    health={}
    for name,values in (("v2",a),("lazy",b)):
        health[name]=dict(nan=sum(int(torch.isnan(g).sum()) for g in values.values() if g is not None),
                          inf=sum(int(torch.isinf(g).sum()) for g in values.values() if g is not None))
    return dict(global_statistics=all_stats,modules=modules,strict_tolerance_label=strict_label,
                strict_failed_parameters=failures,strict_failed_parameter_count=len(failures),nonfinite_gradients=health)


def set_reader_arm(model,family,implementation):
    # No global flags change: B and R both use the same IEEE process backend.
    policy=READER_POLICIES[family]
    model.precision_policy=policy;model.forest.precision_policy=policy
    model.forest.implementation=implementation
    for reader in model.readers.values():reader.precision_mode=policy.reader
    return policy
