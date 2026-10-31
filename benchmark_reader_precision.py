"""One-round B1/B2/B1-R/B2-R snapshot benchmark on the same physical model."""
from __future__ import annotations
import argparse,gc,json,statistics,subprocess,sys,time
from pathlib import Path
import torch
import pandas as pd
from benchmark_cky_v2 import CASES,cpu_snapshot,snapshot_hash
from benchmark_precision_v25 import step,restore
from models.model import LatentForestLM
from models.precision_policy import configure_model,backend_state
from reader_precision_study import READER_POLICIES,set_reader_arm,ReaderObserver
from profile_performance import make_batch,staged_step
from train import make_optimizer
from utils import load_config,to_device,set_seed,rng_state,atomic_json,device_info

ARMS=[("B1","B","score_first"),("B2","B","lazy_semantic"),("B1-R","R","score_first"),("B2-R","R","lazy_semantic")]
SELECTED={"smoke_b2_w16","smoke_b2_w24","main_b1_w24","main_b4_w24","real_b2"}


def run_case(case,args):
    name,config,batch_size,words,real=case;set_seed(42)
    cfg=load_config(f"configs/{config}.yaml");cfg["data"]["cache_dir"]=args.cache_dir
    policy=READER_POLICIES["B"]
    model=configure_model(LatentForestLM(cfg["model"]).cuda().train(),policy);opt=make_optimizer(model,cfg["train"]);scaler=policy.make_scaler()
    batch=to_device(make_batch(cfg,batch_size,4,words,128,"cache" if real else "synthetic"),torch.device("cuda"))
    for p in model.parameters():p.grad=torch.zeros_like(p)
    opt.step();opt.zero_grad(set_to_none=True);torch.cuda.synchronize()
    state=dict(model=cpu_snapshot(model.state_dict()),optimizer=cpu_snapshot(opt.state_dict()),scaler=scaler.state_dict(),rng=rng_state())
    sha=snapshot_hash(state["model"])
    def select(arm):
        label,family,impl=arm
        restore(model,opt,scaler,state,impl)
        return set_reader_arm(model,family,impl)
    for arm in ARMS:
        pp=select(arm)
        for _ in range(args.warmup):assert not step(model,batch,opt,pp,scaler)
    samples={a[0]:[] for a in ARMS};peaks={a[0]:[] for a in ARMS};orders=[]
    for repeat in range(args.repeats):
        shift=repeat%len(ARMS);order=ARMS[shift:]+ARMS[:shift]
        if repeat//len(ARMS)%2:order=list(reversed(order))
        orders.append([x[0] for x in order])
        for arm in order:
            pp=select(arm);torch.cuda.reset_peak_memory_stats()
            start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            tick=time.perf_counter();start.record();skipped=step(model,batch,opt,pp,scaler);end.record();end.synchronize()
            assert not skipped
            samples[arm[0]].append(dict(repeat=repeat,wall_ms=(time.perf_counter()-tick)*1000,
                cuda_envelope_ms=start.elapsed_time(end),skipped=False))
            peaks[arm[0]].append(torch.cuda.max_memory_allocated()/2**30)
    rows=[];diagnostics={};types={}
    for arm in ARMS:
        label,family,impl=arm;parts=[]
        for _ in range(3):
            pp=select(arm)
            parts.append(staged_step(model,batch,opt,pp.outer_autocast,measure_reader_backward=True))
        pp=select(arm);observers={}
        for key,reader in model.readers.items():
            observers[key]=ReaderObserver(save_tensors=False);reader.precision_observer=observers[key]
        with pp.outer_context("cuda"):out=model(**batch)
        assert torch.isfinite(out["logits"]).all() and torch.isfinite(out["memory"]["semantic"]).all()
        for reader in model.readers.values():reader.precision_observer=None
        expected="torch.bfloat16" if family=="B" else "torch.float32"
        for obs in observers.values():
            for key in ("query_projection","key_projection","value_projection","attention_weighted_values","output_projection"):
                assert obs.specs[key]["actual_dtype"]==expected,(label,key,obs.specs[key])
            assert obs.specs["attention_probabilities"]["actual_dtype"]=="torch.float32"
            assert obs.backend["policy"]=="math"
        types[label]={key:obs.summary() for key,obs in observers.items()};del out
        pp=select(arm)
        import benchmark_cky_v25 as old
        original=old.regular_step
        old.regular_step=lambda m,bb,o,p:step(m,bb,o,pp,scaler)
        try:counts=old.operator_counts(model,batch,opt,pp.outer_autocast)
        finally:old.regular_step=original
        wall=[v["wall_ms"] for v in samples[label]];tokens=int(batch["labels"].ne(-100).sum())
        row=dict(case=name,arm=label,implementation=impl,reader=pp.reader,outer=pp.outer_autocast,composer=pp.composer,
            batch=batch_size,words_actual=batch["history_ids"].shape[-1],target_length=batch["labels"].shape[-1],ce_tokens=tokens,
            weight_sha256=sha,step_median_ms=statistics.median(wall),step_min_ms=min(wall),step_max_ms=max(wall),
            ce_tokens_per_second=tokens*1000/statistics.median(wall),peak_allocated_gib=max(peaks[label]),samples=len(wall),
            nan_inf_outputs=0,nonfinite_gradient_steps=0,**counts)
        for prefix,key in [("reader_forward","module/reader"),("reader_backward","module/reader_backward"),
                           ("forest_forward","module/cky"),("forest_backward","phase/cky_backward")]:
            values=[p[key].get("wall_ms",p[key]["cuda_envelope_ms"]) for p in parts]
            row.update({prefix+"_median_ms":statistics.median(values),prefix+"_min_ms":min(values),prefix+"_max_ms":max(values)})
        rows.append(row);diagnostics[label]=parts;print(json.dumps(row),flush=True)
    paired={}
    for key,numerator,denominator in [("B_speedup","B1","B2"),("R_speedup","B1-R","B2-R"),("R_over_B_lazy_cost","B2-R","B2")]:
        ratios=[a["wall_ms"]/b["wall_ms"] for a,b in zip(samples[numerator],samples[denominator])]
        paired[key]=dict(median=statistics.median(ratios),minimum=min(ratios),maximum=max(ratios),samples=ratios)
    root=Path(args.output_dir)/"performance"/name;root.mkdir(parents=True,exist_ok=True)
    atomic_json(dict(samples=samples,orders=orders,paired=paired,protocol="All four arms in the same round; restore same model/AdamW/RNG; IEEE global unchanged"),root/"samples.json")
    atomic_json(diagnostics,root/"diagnostics.json");atomic_json(types,root/"dtype_trace.json")
    pd.DataFrame(rows).to_csv(root/"benchmark.csv",index=False)
    del model,opt,state,batch;gc.collect();torch.cuda.empty_cache()
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output-dir",default="reports/cky_v25_reader")
    p.add_argument("--cache-dir",default="cache/tinystories_integration");p.add_argument("--warmup",type=int,default=2);p.add_argument("--repeats",type=int,default=8)
    p.add_argument("--worker",action="store_true");p.add_argument("--case",choices=sorted(SELECTED));p.add_argument("--cases",nargs="+",choices=sorted(SELECTED));p.add_argument("--resume",action="store_true")
    a=p.parse_args()
    if a.warmup<2 or a.repeats<5:p.error("Require warmup>=2 and valid samples>=5")
    root=Path(a.output_dir);(root/"logs").mkdir(parents=True,exist_ok=True)
    if a.worker:
        torch.set_num_threads(4);READER_POLICIES["B"].configure_backend()
        case=next(c for c in CASES if c[0]==a.case);run_case(case,a);return
    atomic_json(dict(environment=device_info(),backend=backend_state(),backend_scope="coordinator startup before worker backend configuration",
        worker_backend_policy="Each worker configures B IEEE before model construction; all four arms retain that same process policy",arguments=vars(a),
        protocol="Balanced rotated/reversed four-arm order; direct same-round B2 versus B2-R; setup common zero-gradient AdamW state initialization"),root/"benchmark_environment.json")
    for case in CASES:
        if case[0] not in SELECTED or (a.cases and case[0] not in a.cases):continue
        existing=root/"performance"/case[0]/"benchmark.csv"
        if a.resume and existing.exists():continue
        cmd=[sys.executable,__file__,"--worker","--case",case[0],"--output-dir",str(root),"--cache-dir",a.cache_dir,"--warmup",str(a.warmup),"--repeats",str(a.repeats)]
        print("Reader benchmark",case[0],flush=True)
        with (root/"logs"/f"benchmark_{case[0]}.log").open("w",encoding="utf-8") as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
        if r.returncode:raise RuntimeError(f"Benchmark {case[0]} failed; see log")
        paths=list((root/"performance").glob("*/benchmark.csv"));pd.concat([pd.read_csv(p) for p in paths]).to_csv(root/"benchmark.csv",index=False)

if __name__=="__main__":main()
