"""Candidate-ID aligned, root-closure-only activation diagnostics."""
from __future__ import annotations
from collections import defaultdict
import torch
from models.composer import Composer
from models.forest import SemanticForest
from models.precision_policy import POLICIES
from utils import restore_rng,atomic_json,atomic_torch_save

from benchmark_cky_v2 import cpu_snapshot
from precision_study import errors,tensor_health


def topology(output,mask):
    spans=output["spans"];n,w=mask.shape
    base={};decoded={};children={};offset=0
    for (i,j),cell in sorted(spans.items(),key=lambda x:(x[0][1]-x[0][0],x[0][0])):
        k=cell["scores"].shape[-1];base[i,j]=offset
        for r in range(n):
            for slot in range(k):decoded[offset+r*k+slot]=(r,i,j,slot)
        offset+=n*k
    for cid,(r,i,j,slot) in decoded.items():
        if j-i==1:continue
        flat=int(spans[i,j]["retained_flat"][r,slot]);off,l,rr=spans[i,j]["metadata"][flat]
        kl=spans[i,i+off]["scores"].shape[-1];kr=spans[i+off,j]["scores"].shape[-1]
        children[cid]=(base[i,i+off]+r*kl+l,base[i+off,j]+r*kr+rr)
    roots=[]
    for r,L in enumerate(mask.sum(-1).tolist()):
        k=spans[0,max(1,L)]["scores"].shape[-1]
        roots.extend(base[0,max(1,L)]+r*k+s for s in range(k) if output["candidate_mask"][r,s])
    def closure(start):
        needed=set();stack=list(start)
        while stack:
            cid=stack.pop()
            if cid in needed:continue
            needed.add(cid);stack.extend(children.get(cid,()))
        return needed
    all_needed=closure(roots);chosen_root=roots[0];selected=closure([chosen_root])
    heights={}
    for cid in sorted(selected):heights[cid]=0 if cid not in children else 1+max(heights[x] for x in children[cid])
    plans={"score_first":defaultdict(list),"lazy_semantic":defaultdict(list)}
    packed=defaultdict(list)
    for cid in sorted(all_needed):
        _,i,j,_=decoded[cid]
        if j-i>1:packed[j-i].append(cid)
    for cid in sorted(selected):
        r,i,j,slot=decoded[cid];width=j-i
        if width==1:continue
        chosen=spans[i,j]["selected_flat"][r].tolist()
        rr=chosen.index(int(spans[i,j]["retained_flat"][r,slot]));budget=len(chosen)
        plans["score_first"][width].append((cid,r*(w-width+1)*budget+i*budget+rr))
        plans["lazy_semantic"][width].append((cid,packed[width].index(cid)))
    return plans,decoded,heights,chosen_root


class Capture:
    def __init__(self,forest,plan):
        self.forest,self.plan,self.values,self.handles=forest,plan,{},[]
    def save(self,name,value):
        entries=self.plan.get(self.forest.composer._forest_width,[])
        if not entries:return
        flat=value.detach().reshape(-1,value.shape[-1])
        indices=torch.tensor([v[1] for v in entries],device=value.device)
        cpu=flat.index_select(0,indices).cpu()
        for q,(cid,_) in enumerate(entries):self.values[cid,name]=cpu[q]
    def __enter__(self):
        c=self.forest.composer;self.forest.precision_diagnostics=True
        for name,module in c.named_modules():
            if name in ("mlp.0","mlp.1","mlp.2","gate","left_residual","right_residual","norm"):
                self.handles.append(module.register_forward_hook(lambda m,i,o,name=name:self.save(name,o)))
        self.handles.append(c.norm.register_forward_pre_hook(lambda m,i:self.save("norm_input",i[0])))
        def observe(module,left,right,gate,residual,result):
            for name,x in [("left_input",left),("right_input",right),("gate_sigmoid",gate),("residual",residual),("parent",result)]:self.save(name,x)
        c.precision_trace=observe
        c.precision_state_trace=lambda m,o:self.save("stored_state",o)
        return self
    def __exit__(self,*args):
        for h in self.handles:h.remove()
        self.forest.composer.precision_trace=None;self.forest.composer.precision_state_trace=None
        self.forest.precision_diagnostics=False


def run_fixed_trace(group,root,export_width=None):
    if group=="S":return dict(status="not repeated",reason="Fixed diagnostic scores are already FP32; same inputs as B")
    policy=POLICIES[group];source=torch.load(root/"fixed_inputs.pt",map_location="cpu",weights_only=False)
    cfg=source["config"]
    f=SemanticForest(Composer(cfg["semantic_size"],cfg["composer_hidden"]),4,12,cfg["prune_delta"],cfg["exploration"]).cuda().train()
    f.composer.load_state_dict(source["composer"]);f.precision_policy=policy
    mask=source["mask"].cuda();inputs=[source[k].cuda().requires_grad_() for k in ("leaves","scores","coeff")]
    restore_rng(source["rng"])
    with torch.no_grad(),policy.outer_context("cuda"):base=f(*inputs,mask,return_spans=True)
    plans,decoded,heights,chosen=topology(cpu_snapshot(base),source["mask"])
    values=[];roots=[];grads=[];dtype=[]
    for impl in ("score_first","lazy_semantic"):
        f.implementation=impl;f.zero_grad(set_to_none=True)
        for x in inputs:x.grad=None
        restore_rng(source["rng"])
        with Capture(f,plans[impl]) as capture,policy.outer_context("cuda"):
            out=f(*inputs,mask)
            probe=torch.linspace(-.5,.5,out["semantic"].numel(),device="cuda").reshape_as(out["semantic"])
            loss=(out["semantic"]*probe).mean()+out["log_prior"].square().mean()*.001
        scaler=policy.make_scaler();scale=scaler.get_scale();scaler.scale(loss).backward()
        grads.append({n:p.grad.detach().cpu()/scale for n,p in f.composer.named_parameters() if p.grad is not None})
        values.append(capture.values);roots.append(out["semantic"].detach().cpu())
    nodes=[]
    for cid in sorted(heights):
        r,i,j,slot=decoded[cid]
        if j-i==1:continue
        components={name:errors(values[0][cid,name],values[1][cid,name]) for cc,name in values[0] if cc==cid}
        nodes.append(dict(candidate_id=cid,row=r,span=[i,j],slot=slot,width=j-i,height=heights[cid],
            dtypes={name:[str(values[a][cid,name].dtype) for a in range(2)] for cc,name in values[0] if cc==cid},
            errors=components,health=tensor_health(values[1][cid,"stored_state"])))
    from validate_cky_v25 import gradient_statistics
    result=dict(status="completed",root_id=chosen,root_errors=errors(*roots),nodes=nodes,
        gradient=gradient_statistics(*grads),note="Fixed IEEE FP32 leaves/potentials/coefficients and exploration RNG; isolated root-probe backward, not LM CE")
    atomic_torch_save(dict(values=values,roots=roots,nodes=nodes),root/f"trace_values_{group}.pt")
    if export_width:
        atomic_torch_save([{k:v for k,v in arm.items() if decoded[k[0]][2]-decoded[k[0]][1]==export_width} for arm in values],root/f"width_{export_width}_{group}.pt")
    atomic_json(result,root/f"layer_trace_{group}.json")
    return result
