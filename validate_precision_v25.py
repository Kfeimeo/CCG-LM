"""Isolated-process A--E precision study; original strict tolerances retained."""
from __future__ import annotations
import argparse,gc,json,subprocess,sys
from pathlib import Path
import torch
from models.precision_policy import POLICIES,configure_model,backend_state
from models.model import LatentForestLM
from utils import atomic_json,atomic_torch_save,set_seed,load_config,rng_state,restore_rng,device_info
from benchmark_cky_v2 import snapshot_hash,cpu_snapshot
from precision_study import errors,gradient_report,search_equal,dtype_trace,margins_observer,tensor_health
from validate_cky_v25 import model_batch


def cpu_output(out):
    mem=out["memory"]
    return dict(memory={k:cpu_snapshot(mem[k]) for k in ("semantic","scores","vector_coeff","log_prior","candidate_mask","spans")},
                logits=out["logits"].detach().cpu(),loss=out["loss"].detach().cpu())


def full_model_case(group,config,mode,root,batch_override=None,words_override=None):
    policy=POLICIES[group];set_seed(42)
    cfg=load_config(f"configs/{config}.yaml");cfg["model"]["mode"]=mode
    model=configure_model(LatentForestLM(cfg["model"]).cuda().train(),policy)
    weight_hash=snapshot_hash(cpu_snapshot(model.state_dict()))
    batch=model_batch(cfg,batch=batch_override if batch_override is not None else (1 if config=="main_125m" else 2),words=words_override if words_override is not None else (24 if config=="main_125m" else 8),
                      histories=4 if config=="main_125m" else 2,target_length=16,device="cuda")
    model.forest.capture_spans=True
    if group=="E" and config=="main_125m":
        ids=batch["history_ids"].reshape(-1,24);mask=batch["word_mask"].reshape(-1,24)
        state=rng_state()
        with torch.no_grad(),policy.outer_context("cuda"):
            leaves=model.semantic_embedding(ids)*mask.unsqueeze(-1);scores,u=model.syntax(ids,mask)
        atomic_torch_save(dict(leaves=leaves.cpu(),scores=scores.cpu(),coeff=u.cpu(),mask=mask.cpu(),
            composer=cpu_snapshot(model.composer.state_dict()),rng=rng_state(),config=cfg["model"]),root/"fixed_inputs.pt")
        restore_rng(state)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3)
    initial=rng_state();outputs=[];grads=[];dtypes=[];ends=[];reader_values=[];margins=[];health=[]
    for impl in ("score_first","lazy_semantic"):
        model.forest.implementation=impl;model.zero_grad(set_to_none=True);restore_rng(initial)
        scaler=policy.make_scaler();rr=[];mm=[]
        handles=[r.register_forward_hook(lambda m,i,o:rr.append(o[0].detach().cpu())) for r in model.readers.values()]
        model.forest.precision_observer=margins_observer(mm)
        with dtype_trace(model) as trace,policy.outer_context("cuda"):
            out=model(**batch)
        ends.append(rng_state())
        scaler.scale(out["loss"]).backward()
        if scaler.is_enabled(): scaler.unscale_(optimizer)
        grads.append({n:None if p.grad is None else p.grad.detach().cpu().clone() for n,p in model.named_parameters()})
        outputs.append(cpu_output(out));dtypes.append(trace);reader_values.append(rr);margins.append(mm)
        health.append(dict(loss=tensor_health(out["loss"]),root=tensor_health(out["memory"]["semantic"]),
                           grad_nan=sum(int(torch.isnan(x).sum()) for x in grads[-1].values() if x is not None),
                           grad_inf=sum(int(torch.isinf(x).sum()) for x in grads[-1].values() if x is not None)))
        for h in handles:h.remove()
        del out
    assert torch.equal(ends[0]["torch"],ends[1]["torch"])
    assert all(torch.equal(x,y) for x,y in zip(ends[0]["cuda"],ends[1]["cuda"]))
    structural=search_equal(outputs[0]["memory"],outputs[1]["memory"]);assert structural["identical"]
    forward={k:errors(outputs[0]["memory"][k],outputs[1]["memory"][k]) for k in ("semantic","scores","vector_coeff","log_prior")}
    forward.update(logits=errors(outputs[0]["logits"],outputs[1]["logits"]),loss=errors(outputs[0]["loss"],outputs[1]["loss"]))
    forward["readers"]=[errors(a,b) for a,b in zip(*reader_values)]
    gradients=gradient_report(*grads)
    row=dict(group=group,config=config,mode=mode,batch=batch["input_ids"].shape[0],words=batch["history_ids"].shape[-1],policy=policy.state_dict(),weight_sha256=weight_hash,
             forward=forward,gradients=gradients,structure=structural,rng_identical=True,health=health,
             parameter_dtypes=sorted({str(p.dtype) for p in model.parameters()}),dtypes=dtypes,margins=margins)
    if config=="main_125m":
        if group=="E": atomic_torch_save(dict(outputs=outputs,grads=grads[0]),root/"fp32_reference.pt")
        else:
            baseline=torch.load(root/"fp32_reference.pt",map_location="cpu",weights_only=False)
            row["cross_precision_vs_E"]=dict(note="Joint outer/Composer precision effect, not Lazy error; root semantic only if search identical",
                structure=[search_equal(baseline["outputs"][i]["memory"],outputs[i]["memory"]) for i in range(2)],
                loss=[errors(baseline["outputs"][i]["loss"],outputs[i]["loss"]) for i in range(2)],
                logits=[errors(baseline["outputs"][i]["logits"],outputs[i]["logits"]) for i in range(2)])
            del baseline
    del model,optimizer,grads,outputs;gc.collect();torch.cuda.empty_cache()
    return row


def matrix(group):
    import itertools
    import validate_cky_v25 as old
    from models.forest import SemanticForest
    policy=POLICIES[group]
    class PolicyForest(SemanticForest):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs);self.precision_policy=policy
    old.SemanticForest=PolicyForest
    rows=[]
    for b,w,k,r,vector in itertools.product([1,2,4],[4,8,16,24],[1,2,4],[4,12],[False,True]):
        for p in ("normal","explore"):
            rows.append(old.forest_case(b,w,k,r,vector,"cuda",policy.outer_autocast,p))
        if len(rows)%48==0:print(group,"matrix",len(rows),"/288",flush=True)
    edges=[]
    for p in ("prune","ties","unreachable","empty","partial_inf"):
        for vector in (False,True):edges.append(old.forest_case(2,8,4,12,vector,"cuda",policy.outer_autocast,p))
    return dict(cases=rows,edge_cases=edges,status="passed")


def worker(args):
    torch.set_num_threads(1);policy=POLICIES[args.group];policy.configure_backend()
    root=Path(args.output_dir);root.mkdir(parents=True,exist_ok=True)
    result=dict(group=args.group,policy=policy.state_dict(),environment=device_info(),backend=backend_state())
    if not args.skip_matrix:result["matrix"]=matrix(args.group)
    result["models"]=[full_model_case(args.group,"smoke","forest_scalar4",root),
                      full_model_case(args.group,"main_125m","forest_vector4",root)]
    from precision_trace import run_fixed_trace
    result["fixed_trace"]=run_fixed_trace(args.group,root,args.export_width)
    atomic_json(result,root/f"numerical_{args.group}.json")
    print(args.group,"completed",flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output-dir",default="reports/cky_v25_precision")
    p.add_argument("--groups",nargs="+",default=["E","A","B","C","D","S"],choices=list(POLICIES))
    p.add_argument("--worker",action="store_true");p.add_argument("--group",choices=list(POLICIES))
    p.add_argument("--skip-matrix",action="store_true");p.add_argument("--resume",action="store_true")
    p.add_argument("--export-width",type=int);args=p.parse_args()
    if args.worker:return worker(args)
    root=Path(args.output_dir);(root/"logs").mkdir(parents=True,exist_ok=True)
    for group in args.groups:
        missing_cases=any(not (root/f"numerical_{g}.json").exists() for g in args.groups if g!="E")
        regenerate_reference=group=="E" and missing_cases and not (root/"fp32_reference.pt").exists()
        if args.resume and (root/f"numerical_{group}.json").exists() and not regenerate_reference:continue
        cmd=[sys.executable,__file__,"--worker","--group",group,"--output-dir",str(root)]
        if args.skip_matrix or group=="S":cmd.append("--skip-matrix")
        if args.export_width:cmd.extend(["--export-width",str(args.export_width)])
        print("Numerical worker",group,flush=True)
        with (root/"logs"/f"validate_{group}.log").open("w",encoding="utf-8") as f:
            r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
        if r.returncode:raise RuntimeError(f"Numerical {group} failed; see log")
        all_results=[json.loads(x.read_text(encoding="utf-8")) for x in root.glob("numerical_?.json")]
        atomic_json(all_results,root/"numerical_comparison.json")
        atomic_json([{k:x[k] for k in ("group","models")} for x in all_results],root/"gradient_comparison.json")
        atomic_json([{k:m[k] for k in ("config","mode","dtypes","parameter_dtypes")} | {"group":x["group"]}
                     for x in all_results for m in x["models"]],root/"dtype_trace.json")

if __name__=="__main__": main()
