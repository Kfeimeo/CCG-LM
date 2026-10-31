"""Shared numerical/dtype instrumentation, never enabled in timed training."""
from __future__ import annotations
import math
from collections import defaultdict
from contextlib import contextmanager
import torch

GROUPS={
 "syntax_encoder":lambda n:n.startswith("syntax.") and not n.startswith("syntax.span_scorer."),
 "span_scorer":lambda n:n.startswith("syntax.span_scorer."),
 "semantic_embedding":lambda n:n.startswith("semantic_embedding."),
 "composer":lambda n:n.startswith("forest.composer."),
 "vector_coefficient_head":lambda n:n.startswith("syntax.span_scorer.vector."),
 "semantic_reader":lambda n:n.startswith("readers."),
 "decoder":lambda n:n.startswith("decoder."),
}

def errors(a,b):
    x,y=a.detach().float().cpu(),b.detach().float().cpu()
    diff=x-y
    finite=torch.isfinite(x)&torch.isfinite(y)
    if not finite.all(): return dict(nonfinite_reference=int((~torch.isfinite(x)).sum()),nonfinite_actual=int((~torch.isfinite(y)).sum()))
    if not x.numel(): return dict(max_absolute_error=0.,normalized_l2_error=0.,rms_error=0.,reference_norm=0.,actual_norm=0.)
    norm=float(x.double().norm())
    return dict(max_absolute_error=float(diff.abs().max()),normalized_l2_error=float(diff.double().norm())/max(norm,1e-30),
        rms_error=float(diff.double().square().mean().sqrt()),reference_norm=norm,actual_norm=float(y.double().norm()))

def gradient_report(a,b):
    from validate_cky_v25 import gradient_statistics
    result={};violations=[]
    for group,predicate in GROUPS.items():
        ga={n:v for n,v in a.items() if predicate(n)};gb={n:b[n] for n in ga}
        if not ga: result[group]=dict(status="absent");continue
        # Original V2 mixed-precision tolerances remain unchanged.
        stat=gradient_statistics(ga,gb)
        stat.update(grad_norm_reference=stat["gradient_norm_reference"],grad_norm_lazy=stat["gradient_norm_lazy"],
            max_absolute_error=stat["max_absolute_gradient_error"],normalized_l2_error=stat["normalized_l2_gradient_error"],
            significant_gradient_relative_error=stat["max_relative_gradient_error_above_1e_5"])
        stat["strict_bf16_failures"]=[];stat["strict_fp32_failures"]=[]
        for name,g in ga.items():
            if g is None: continue
            for label,atol,rtol in [("bf16",.00015,.04),("fp32",3e-6,.0003)]:
                try: torch.testing.assert_close(g,gb[name],atol=atol,rtol=rtol)
                except AssertionError: stat["strict_"+label+"_failures"].append(name)
        result[group]=stat
    return result

def tensor_health(tensor):
    x=tensor.detach().float()
    finite=torch.isfinite(x);ax=x.abs();nz=ax[(ax>0)&finite]
    return dict(dtype=str(tensor.dtype),nan=int(torch.isnan(x).sum()),inf=int(torch.isinf(x).sum()),
        zeros=int(x.eq(0).sum()),elements=x.numel(),minimum_nonzero=float(nz.min()) if nz.numel() else None,
        max_absolute=float(ax[finite].max()) if finite.any() else None,
        fp16_subnormal_range=int(((ax>0)&(ax<torch.finfo(torch.float16).tiny)).sum()))

@contextmanager
def dtype_trace(model):
    rows=[];handles=[]
    names={"semantic_embedding","syntax","syntax.span_scorer","forest","forest.composer",
           "forest.composer.mlp.0","forest.composer.mlp.1","forest.composer.mlp.2","forest.composer.gate",
           "forest.composer.left_residual","forest.composer.right_residual","forest.composer.norm"}
    def describe(x):
        if isinstance(x,torch.Tensor): return dict(shape=list(x.shape),dtype=str(x.dtype))
        if isinstance(x,(tuple,list)): return [describe(v) for v in x]
        if isinstance(x,dict): return {k:describe(v) for k,v in x.items() if k in ("semantic","scores","vector_coeff","log_prior")}
        return None
    for name,module in model.named_modules():
        if name not in names and not (name.startswith("readers.") and name.count(".")<=2): continue
        def hook(mod,inp,out,name=name):
            rows.append(dict(module=name,width=getattr(model.composer,"_forest_width",None),
                             input=describe(inp),output=describe(out),
                             weight_dtype=str(mod.weight.dtype) if hasattr(mod,"weight") else None))
        handles.append(module.register_forward_hook(hook))
    model.forest.precision_diagnostics=True
    try: yield rows
    finally:
        for h in handles:h.remove()
        model.forest.precision_diagnostics=False


def search_equal(a,b):
    assert a["spans"].keys()==b["spans"].keys()
    changed=0;total=0
    for span,x in a["spans"].items():
        y=b["spans"][span]
        assert x["metadata"]==y["metadata"]
        for key in ("selected_flat","retained_flat","candidate_mask","scores"):
            if x[key] is None: continue
            changed+=int(x[key].detach().cpu().ne(y[key].detach().cpu()).sum());total+=x[key].numel()
    return dict(changed_entries=changed,entries=total,identical=changed==0)


def margins_observer(rows):
    def record(width,scores,mask,k):
        if scores.shape[-1]<=k:
            rows.append(dict(width=width,status="no_k_plus_one"));return
        top=scores.detach().masked_fill(~mask,-torch.inf).topk(k+1,-1).values
        x=(top[...,k-1]-top[...,k]).cpu();finite=x[torch.isfinite(x)]
        rows.append(dict(width=width,count=finite.numel(),minimum=float(finite.min()) if finite.numel() else None,
                         median=float(finite.median()) if finite.numel() else None,
                         below_1e_5=int((finite<1e-5).sum()),below_1e_3=int((finite<1e-3).sum()),
                         note="pre-exploration Top-K margin in selected Top-R pool"))
    return record
