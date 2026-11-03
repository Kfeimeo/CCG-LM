"""Pure-tensor V3 width step for the V3.5 torch.compile experiments.

The eager `lazy_tensorized` loop in forest_tensorized.py is unchanged and stays the
reference. When a forest's `search_backend` is 'function', 'compile_default' or
'compile_reduce_overhead', `_search_scores` calls `width_step` (plain or compiled)
with the very same operation sequence. Python scalars (spans, candidates, budget,
keep, explore, swap, delta, exploration) are graph constants, so each width of each
(rows, words) shape is its own graph; recompilation counts are reported, never hidden.

Inductor settings forced before any compilation (all three are required for the
search identity to hold against eager):
  fallback_random=True          -> eager Philox kernels for rand_like (RNG stream parity)
  emulate_precision_casts=True  -> the width-2 BF16 leaf+leaf add is rounded to BF16 as in eager
  recompile_limit raised        -> 23 widths x shapes exceed the default of 8

Rows (dim 0 of bank / mask_bank / span_flat / left_ids / right_ids) are marked dynamic (DYNAMIC_ROWS): real training
truncates the last micro-batch of every update to the exact token budget, so the row count
changes every update and a fully static graph per (rows, words, width) exceeded the recompile
limit after six updates (Stage E, first attempt; recorded in the report). The arithmetic of a
width step is row-wise independent, so a symbolic row count changes neither the operations
nor the RNG consumption order; every other dimension stays static (automatic dynamic off).
"""
from __future__ import annotations
import torch

COMPILE_BACKENDS=('eager','function','compile_default','compile_reduce_overhead')
DYNAMIC_ROWS=True
_compiled={}


def mark_rows_dynamic(*tensors):
    if DYNAMIC_ROWS:
        for t in tensors:torch._dynamo.mark_dynamic(t,0)


def width_step(bank,mask_bank,span_flat,left,right,parent,left_ids,right_ids,spans,candidates,budget,keep,explore,swap,delta,exploration):
    n=bank.shape[0]
    ls=bank.index_select(1,left.reshape(-1)).reshape(n,spans,candidates)
    rs=bank.index_select(1,right.reshape(-1)).reshape(n,spans,candidates)
    lm=mask_bank.index_select(1,left.reshape(-1)).reshape(n,spans,candidates)
    rm=mask_bank.index_select(1,right.reshape(-1)).reshape(n,spans,candidates)
    local=span_flat.index_select(1,parent)
    score=(ls+rs).float()+local.unsqueeze(-1)
    valid=(lm&rm)&torch.isfinite(score);score=score.masked_fill(~valid,-torch.inf)
    priority=score.detach();chosen=priority.topk(budget-explore,dim=-1).indices
    if explore:
        noise=torch.rand_like(priority).masked_fill(~valid,-torch.inf);noise.scatter_(2,chosen,-torch.inf)
        chosen=torch.cat((chosen,noise.topk(explore,dim=-1).indices),-1)
    pair_mask=valid.gather(2,chosen);expanded_score=score.gather(2,chosen)
    retained_score,rank=expanded_score.topk(keep,dim=-1)
    if swap:
        eligible=pair_mask.clone();eligible.scatter_(2,rank[...,:-1],False)
        if delta is not None:eligible&=expanded_score>=retained_score[...,:1]-delta
        random_priority=torch.rand_like(expanded_score).masked_fill(~eligible,-torch.inf)
        random_rank=random_priority.argmax(-1,keepdim=True)
        use_random=(torch.rand_like(retained_score[...,:1])<exploration)&eligible.any(-1,keepdim=True)
        last=torch.where(use_random,random_rank,rank[...,-1:]);rank=torch.cat((rank[...,:-1],last),-1)
        retained_score=expanded_score.gather(2,rank)
    retained_mask=pair_mask.gather(2,rank)
    if delta is not None:retained_mask=retained_mask&(retained_score>=retained_score[...,:1]-delta)
    retained_score=retained_score.masked_fill(~retained_mask,0.)
    retained_flat=chosen.gather(2,rank)
    lp=left_ids.gather(2,retained_flat);rp=right_ids.gather(2,retained_flat)
    return retained_score,retained_mask,lp,rp,chosen,retained_flat,valid


def configure_inductor(recompile_limit=1024):
    import torch._dynamo
    import torch._inductor.config as inductor
    inductor.fallback_random=True
    inductor.emulate_precision_casts=True
    torch._dynamo.config.automatic_dynamic_shapes=False  # only the explicitly marked rows dim is symbolic
    torch._dynamo.config.recompile_limit=recompile_limit
    torch._dynamo.config.accumulated_recompile_limit=max(torch._dynamo.config.accumulated_recompile_limit,4*recompile_limit)
    return dict(fallback_random=inductor.fallback_random,emulate_precision_casts=inductor.emulate_precision_casts,
                recompile_limit=torch._dynamo.config.recompile_limit,accumulated_recompile_limit=torch._dynamo.config.accumulated_recompile_limit,
                cudagraph_trees=inductor.triton.cudagraph_trees,dynamic_rows=DYNAMIC_ROWS,automatic_dynamic_shapes=torch._dynamo.config.automatic_dynamic_shapes)


def begin_step(backend):
    """Mark a new CUDA-graph-trees generation before a compiled search forward (no-op otherwise)."""
    if backend=='compile_reduce_overhead':torch.compiler.cudagraph_mark_step_begin()


def get_width_step(backend):
    if backend not in COMPILE_BACKENDS:raise ValueError(f'Unknown search backend {backend}')
    if backend in ('eager','function'):return width_step
    if backend not in _compiled:
        configure_inductor()
        mode={'compile_default':'default','compile_reduce_overhead':'reduce-overhead'}[backend]
        # reduce-overhead uses PyTorch's default CUDA-graph *trees* (the non-tree path needs fully
        # static shapes and would re-specialise the symbolic rows dim, see DYNAMIC_ROWS). Trees
        # invalidate a previous step's graph outputs once a new step begins; with V3 (grad) and
        # V3-C (no_grad) alternating in one process this raised "accessing tensor output of
        # CUDAGraphs that has been overwritten by a subsequent run" inside the width loop, so every
        # search forward calls begin_step() = torch.compiler.cudagraph_mark_step_begin() first (the
        # documented fix) and callers must not read a compiled forward's outputs after the next one.
        # dynamic=None: static by default (assume_static_by_default), symbolic only where mark_dynamic says so.
        _compiled[backend]=torch.compile(width_step,mode=mode,dynamic=None if DYNAMIC_ROWS else False,fullgraph=True)
    return _compiled[backend]


def compile_counters():
    """Dynamo/inductor counters relevant to the compile experiments (snapshot copies)."""
    from torch._dynamo.utils import counters
    out={}
    for group in ('frames','graph_break','inductor','stats','unimplemented','recompiles'):
        if group in counters:out[group]={k:int(v) for k,v in counters[group].items()}
    return out


def reset_compile_state():
    import torch._dynamo
    torch._dynamo.reset();_compiled.clear()
    from torch._dynamo.utils import counters
    counters.clear()
