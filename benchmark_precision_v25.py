"""Snapshot-paired precision benchmark. Each backend policy owns a subprocess."""
from __future__ import annotations
import argparse,gc,json,math,statistics,subprocess,sys,time
from pathlib import Path
import pandas as pd
import torch
from benchmark_cky_v2 import CASES,cpu_snapshot,snapshot_hash
from models.model import LatentForestLM
from models.precision_policy import POLICIES,configure_model,backend_state
from precision_study import dtype_trace
from profile_performance import make_batch,staged_step
from train import make_optimizer
from utils import atomic_json,set_seed,load_config,rng_state,restore_rng,to_device,device_info


def step(model,batch,opt,policy,scaler):
    opt.zero_grad(set_to_none=True)
    with policy.outer_context("cuda"):out=model(**batch)
    scaler.scale(out["loss"]).backward()
    if scaler.is_enabled():scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
    before=scaler.get_scale();scaler.step(opt);scaler.update()
    return scaler.get_scale()<before


def restore(model,opt,scaler,state,impl):
    opt.zero_grad(set_to_none=True);model.load_state_dict(state["model"]);opt.load_state_dict(state["optimizer"])
    scaler.load_state_dict(state["scaler"]);restore_rng(state["rng"]);model.forest.implementation=impl;torch.cuda.synchronize()


def run_case(case,args):
    name,config,b,words,real=case;policy=POLICIES[args.group];set_seed(42)
    cfg=load_config(f"configs/{config}.yaml");cfg["data"]["cache_dir"]=args.cache_dir
    model=configure_model(LatentForestLM(cfg["model"]).cuda().train(),policy);opt=make_optimizer(model,cfg["train"]);scaler=policy.make_scaler()
    batch=to_device(make_batch(cfg,b,4,words,128,"cache" if real else "synthetic"),torch.device("cuda"))
    # Identical FP32 zero-gradient AdamW initialization, independent of AMP policy.
    # This initializes optimizer buffers once. It is not a measured training update.
    for p in model.parameters():p.grad=torch.zeros_like(p)
    opt.step();opt.zero_grad(set_to_none=True);torch.cuda.synchronize()
    state=dict(model=cpu_snapshot(model.state_dict()),optimizer=cpu_snapshot(opt.state_dict()),scaler=scaler.state_dict(),rng=rng_state())
    weight_hash=snapshot_hash(state["model"])
    for impl in ("score_first","lazy_semantic"):
        restore(model,opt,scaler,state,impl)
        for _ in range(args.warmup):
            if step(model,batch,opt,policy,scaler):raise RuntimeError("FP16 warmup skipped: lower a common scale explicitly and rerun both arms")
    samples={k:[] for k in ("score_first","lazy_semantic")};peaks={k:[] for k in samples}
    for repeat in range(args.repeats):
        for impl in (tuple(samples) if repeat%2==0 else tuple(reversed(samples))):
            restore(model,opt,scaler,state,impl);torch.cuda.reset_peak_memory_stats()
            a,b_event=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            tick=time.perf_counter();a.record();skipped=step(model,batch,opt,policy,scaler);b_event.record();torch.cuda.synchronize()
            if skipped:raise RuntimeError("Skipped update is not an effective benchmark sample")
            samples[impl].append(dict(wall_ms=(time.perf_counter()-tick)*1000,cuda_envelope_ms=a.elapsed_time(b_event),
                                     scaler_scale=scaler.get_scale(),skipped=False))
            peaks[impl].append(torch.cuda.max_memory_allocated()/2**30)
    rows=[];details={};root=Path(args.output_dir)/args.group/name;root.mkdir(parents=True,exist_ok=True)
    for impl in samples:
        diagnostics=[]
        for _ in range(3):
            restore(model,opt,scaler,state,impl)
            diagnostics.append(staged_step(model,batch,opt,policy.outer_autocast,scaler=scaler,measure_composer_backward=True))
        restore(model,opt,scaler,state,impl);shapes=[]
        handle=model.composer.register_forward_pre_hook(lambda m,i:shapes.append(list(i[0].shape)))
        state_health=[]
        model.composer.precision_state_trace=lambda m,o:state_health.append(torch.stack((torch.isnan(o).sum(),torch.isinf(o).sum())))
        with dtype_trace(model) as traces,policy.outer_context("cuda"):out=model(**batch)
        model.composer.precision_state_trace=None
        nan_inf_count=int(torch.stack(state_health).sum()) if state_health else 0
        assert nan_inf_count==0
        model.forest.precision_diagnostics=False;handle.remove()
        actual={x["output"]["dtype"] for x in traces if x["module"]=="forest.composer.mlp.0"}
        expected={"A":"torch.bfloat16","B":"torch.float32","C":"torch.float32","D":"torch.float16","E":"torch.float32","S":"torch.float32"}[args.group]
        assert actual=={expected},(args.group,actual,expected)
        finite=bool(torch.isfinite(out["logits"]).all() and torch.isfinite(out["memory"]["semantic"]).all())
        assert finite
        del out
        restore(model,opt,scaler,state,impl)
        import benchmark_cky_v25 as old
        original=old.regular_step
        old.regular_step=lambda m,bb,o,p:step(m,bb,o,policy,scaler)
        try:counts=old.operator_counts(model,batch,opt,policy.outer_autocast,capture_kernel_names=True)
        finally:old.regular_step=original
        kernels=counts.pop("cuda_kernel_names")
        wall=[s["wall_ms"] for s in samples[impl]];tokens=int(batch["labels"].ne(-100).sum())
        row=dict(group=args.group,case=name,implementation=impl,policy=policy.composer,outer=policy.outer_autocast,
            matmul=policy.matmul_precision,score_dtype=policy.score_dtype,batch=batch["input_ids"].shape[0],
            words_actual=batch["history_ids"].shape[-1],target_length=batch["labels"].shape[-1],ce_tokens=tokens,
            weight_sha256=weight_hash,step_median_ms=statistics.median(wall),step_min_ms=min(wall),step_max_ms=max(wall),
            ce_tokens_per_second=tokens*1000/statistics.median(wall),peak_allocated_gib=max(peaks[impl]),
            composer_rows=sum(math.prod(s[:-1]) for s in shapes),composer_calls=len(shapes),
            nan_inf_count=nan_inf_count,scaler_scale=scaler.get_scale(),skipped_steps=0,actual_linear_dtype=next(iter(actual)),**counts)
        for label,key in [("composer_forward","module/composer"),("composer_backward","module/composer_backward"),
                          ("forest_forward","module/cky"),("forest_backward","phase/cky_backward")]:
            times=[d[key].get("wall_ms",d[key]["cuda_envelope_ms"]) for d in diagnostics]
            row.update({label+"_median_ms":statistics.median(times),label+"_min_ms":min(times),label+"_max_ms":max(times)})
        rows.append(row);details[impl]=dict(samples=samples[impl],diagnostics=diagnostics,dtypes=traces,cuda_gemm_kernel_names=kernels)
        print(json.dumps(row),flush=True)
    pd.DataFrame(rows).to_csv(root/"benchmark.csv",index=False);atomic_json(details,root/"diagnostics.json")
    del model,opt,state,batch;gc.collect();torch.cuda.empty_cache()
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output-dir",default="reports/cky_v25_precision/performance")
    p.add_argument("--cache-dir",default="cache/tinystories_integration");p.add_argument("--groups",nargs="+",choices=list(POLICIES),default=list("ABCDE"))
    p.add_argument("--group",choices=list(POLICIES));p.add_argument("--worker",action="store_true");p.add_argument("--resume",action="store_true")
    p.add_argument("--warmup",type=int,default=2);p.add_argument("--repeats",type=int,default=5)
    p.add_argument("--cases",nargs="+",choices=[c[0] for c in CASES]);args=p.parse_args()
    if args.warmup<2 or args.repeats<5:p.error("Require warmup >=2 and effective samples >=5")
    root=Path(args.output_dir);(root/"logs").mkdir(parents=True,exist_ok=True)
    if args.worker:
        torch.set_num_threads(4);POLICIES[args.group].configure_backend();rows=[]
        atomic_json(dict(environment=device_info(),backend=backend_state(),policy=POLICIES[args.group].state_dict(),arguments=vars(args)),root/f"environment_{args.group}.json")
        for case in CASES:
            if args.cases and case[0] not in args.cases:continue
            existing=root/args.group/case[0]/"benchmark.csv"
            if args.resume and existing.exists():rows.extend(pd.read_csv(existing).to_dict("records"));continue
            rows.extend(run_case(case,args));pd.DataFrame(rows).to_csv(root/f"benchmark_{args.group}.csv",index=False)
        return
    for group in args.groups:
        cmd=[sys.executable,__file__,"--worker","--group",group,"--output-dir",str(root),"--cache-dir",args.cache_dir,
             "--warmup",str(args.warmup),"--repeats",str(args.repeats)]
        if args.resume:cmd.append("--resume")
        if args.cases:cmd.extend(["--cases",*args.cases])
        print("Precision benchmark",group,flush=True)
        with (root/"logs"/f"benchmark_{group}.log").open("w",encoding="utf-8") as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
        if r.returncode:raise RuntimeError(f"Benchmark {group} failed; see log")
        paths=list(root.glob("benchmark_?.csv"));pd.concat([pd.read_csv(x) for x in paths]).to_csv(root.parent/"benchmark.csv",index=False)

if __name__=="__main__":main()
