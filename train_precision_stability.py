"""100-step screen, then selected 500-step / 3-seed smoke stability runs."""
from __future__ import annotations
import argparse,csv,hashlib,json,math,subprocess,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from data.dataset import StoryDataset,StatefulSampler,Collator
from models.model import LatentForestLM
from models.precision_policy import POLICIES,configure_model,backend_state
from train import make_optimizer,make_schedule,train_update,save_checkpoint,load_checkpoint
from utils import atomic_json,load_config,set_seed,to_device,device_info


class FixedSubset:
    """Select complete natural examples; never truncate sentences or targets."""
    def __init__(self,source,max_words=12,max_target=24):
        self.source=source
        self.indices=[i for i in range(len(source)) if source[i]["history"] and
            max(map(len,source[i]["history"]))<=max_words and len(source[i]["labels"])<=max_target]
        self.bpe_offsets=np.cumsum([0]+[len(source[i]["labels"]) for i in self.indices])
        self.fingerprint=hashlib.sha256((source.manifest["fingerprint"]+json.dumps(self.indices)).encode()).hexdigest()
        if not self.indices:raise ValueError("Empty fixed training subset")
    def __len__(self):return len(self.indices)
    def __getitem__(self,i):return self.source[self.indices[i]]


def validation(model,loader,probe,policy):
    model.eval();loss=tokens=roots=histories=entropy=0.
    with torch.no_grad():
        for batch in loader:
            batch=to_device(batch,torch.device("cuda"))
            with policy.outer_context("cuda"):out=model(**batch)
            loss+=float(out["loss_sum"]);tokens+=int(out["target_mask"].sum()) if "target_mask" in out else int(batch["labels"].ne(-100).sum())
            mem=out["memory"];m=mem["candidate_mask"];lp=mem["log_prior"]
            roots+=int(m.sum());histories+=int(m.any(-1).sum());entropy+=float((-(lp.exp()*lp)*m).sum())
        with policy.outer_context("cuda"):pr=model(**probe,return_trees=True)
    model.train();nll=loss/tokens
    return dict(val_nll=nll,val_perplexity=math.exp(nll),val_target_tokens=int(tokens),
        root_candidate_count=roots/max(histories,1),candidate_entropy=entropy/max(histories,1),
        probe_trees=pr["memory"]["trees"])


def worker(args):
    group,impl=args.arm[0],("score_first" if args.arm[1]=="1" else "lazy_semantic")
    policy=POLICIES[group];policy.configure_backend();torch.set_num_threads(4);set_seed(args.seed)
    cfg=load_config(args.config);cfg["seed"]=args.seed;cfg["data"]["cache_dir"]=args.cache_dir;cfg["precision"]=policy.state_dict();cfg["model"]["forest_implementation"]=impl
    cfg["train"].update(effective_tokens=args.effective_tokens,micro_batch_size=4)
    selection=cfg.get("study",{}).get("complete_example_filter",{})
    train=FixedSubset(StoryDataset(args.cache_dir),selection.get("max_history_sentence_words",12),selection.get("max_target_tokens",24));dev=StoryDataset(args.cache_dir,"dev")
    cfg.setdefault("study",{}).update(source_cache_fingerprint=train.source.manifest["fingerprint"],requested_optimizer_steps=args.steps,stage=args.stage)
    cfg["train"]["budget_tokens"]=args.steps*args.effective_tokens
    loader=DataLoader(dev,batch_size=8,shuffle=False,collate_fn=Collator(),num_workers=0,generator=torch.Generator().manual_seed(0))
    probe_examples=[dev[i] for i in range(len(dev)) if dev[i]["history"] and max(map(len,dev[i]["history"]))<=12][:2]
    probe=to_device(Collator()(probe_examples),torch.device("cuda"))
    model=configure_model(LatentForestLM(cfg["model"],args.seed).cuda(),policy)
    optimizer=make_optimizer(model,cfg["train"]);schedule=make_schedule(optimizer,cfg["train"]);sampler=StatefulSampler(train,args.seed);scaler=policy.make_scaler()
    root=Path(args.output_dir)/args.stage/f"{args.arm}_seed{args.seed}";root.mkdir(parents=True,exist_ok=True)
    state=dict(step=0,attempt=0,target_tokens=0,train_seconds=0.,cuda_envelope_ms=0.,skipped_steps=0,clipped_steps=0)
    rows=[];vals=[]
    if args.resume_from:
        saved=load_checkpoint(args.resume_from,model,optimizer,schedule,sampler,cfg,train.fingerprint,scaler=scaler)
        state=saved["state"];rows=saved["log_rows"] or [];vals=state.pop("validations",[])
    metadata=dict(config=cfg,environment=device_info(),backend=backend_state(),training_sample_ids=train.indices,
        training_examples=len(train),training_ce_tokens=int(train.bpe_offsets[-1]),validation_examples=len(dev),
        dataset_fingerprint=train.fingerprint,source_cache_fingerprint=train.source.manifest["fingerprint"],
        note="Untruncated real examples with nonempty history, all history sentences <=12 words and targets <=24 tokens. Full fixed dev split. Smoke model unchanged.")
    atomic_json(metadata,root/"metadata.json")
    if not vals:vals.append(dict(step=state["step"],target_tokens=state["target_tokens"],**validation(model,loader,probe,policy)))
    last_roots=[]
    def root_metrics(module,inp,out):
        if not module.training:return
        m=out["candidate_mask"];lp=out["log_prior"]
        last_roots.append(torch.stack((m.sum().float(),m.any(-1).sum().float(),(-(lp.exp()*lp)*m).sum().float())).detach())
    handle=model.forest.register_forward_hook(root_metrics)
    progress=tqdm(total=args.steps,initial=state["step"],desc=f"{args.arm} seed{args.seed}",ascii=True,mininterval=2)
    while state["step"]<args.steps:
        if state["attempt"]>args.steps*2+50:raise RuntimeError("Too many skipped attempts; report effective updates separately")
        last_roots.clear();torch.cuda.synchronize();a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        tick=time.perf_counter();a.record()
        metrics=train_update(model,sampler,optimizer,schedule,args.effective_tokens,4,torch.device("cuda"),policy.outer_autocast,
                             cfg["train"]["grad_clip"],scaler=scaler)
        b.record();b.synchronize();seconds=time.perf_counter()-tick;cuda_ms=a.elapsed_time(b)
        skipped=metrics.get("skipped_step",False);state["attempt"]+=1;state["step"]+=int(not skipped)
        state["target_tokens"]+=args.effective_tokens;state["train_seconds"]+=seconds;state["cuda_envelope_ms"]+=cuda_ms;state["skipped_steps"]+=int(skipped)
        state["clipped_steps"]+=int(metrics["grad_norm"] is not None and metrics["grad_norm"]>cfg["train"]["grad_clip"])
        rm=torch.stack(last_roots).sum(0).cpu().tolist();r,h,e=rm
        metrics.pop("scaler_scale",None);metrics.pop("skipped_step",None)
        row=dict(arm=args.arm,seed=args.seed,step=state["step"],attempt=state["attempt"],target_tokens=state["target_tokens"],
            **metrics,scaler_scale=scaler.get_scale(),skipped_step=skipped,step_seconds=seconds,cuda_envelope_ms=cuda_ms,
            accumulated_train_seconds=state["train_seconds"],accumulated_cuda_envelope_ms=state["cuda_envelope_ms"],
            clip_fraction=state["clipped_steps"]/state["attempt"],train_root_candidates=r/max(h,1),train_candidate_entropy=e/max(h,1))
        rows.append(row)
        if not skipped:progress.update(1)
        if not skipped and (state["step"]%args.eval_every==0 or state["step"]==args.steps):
            val=dict(step=state["step"],target_tokens=state["target_tokens"],**validation(model,loader,probe,policy));vals.append(val)
            pd.DataFrame(rows).to_csv(root/"training.csv",index=False);atomic_json(vals,root/"validation.json")
            state["validations"]=vals
            save_checkpoint(root/"checkpoint.pt",model,optimizer,schedule,sampler,cfg,state,train.fingerprint,rows,scaler=scaler)
            state.pop("validations",None)
            print(json.dumps(dict(arm=args.arm,seed=args.seed,step=state["step"],val_nll=val["val_nll"],skipped=state["skipped_steps"])),flush=True)
        progress.set_postfix(nll=f"{metrics['train_nll']:.3f}" if metrics["train_nll"] is not None else "nonfinite",skips=state["skipped_steps"])
    progress.close();handle.remove();atomic_json(state,root/"state.json")


def aggregate(root):
    records=[]
    for path in root.glob("*/*/training.csv"):
        frame=pd.read_csv(path);stage=path.parent.parent.name;frame["stage"]=stage;records.append(frame)
    if records:pd.concat(records).to_csv(root.parent/"stability.csv",index=False)
    validations=[]
    for path in root.glob("*/*/validation.json"):
        arm_seed=path.parent.name;arm,seed=arm_seed.split("_seed");stage=path.parent.parent.name
        for x in json.loads(path.read_text(encoding="utf-8")):validations.append(dict(arm=arm,seed=int(seed),stage=stage,**x))
    baseline={(x["stage"],x["seed"],x["step"]):x for x in validations if x["arm"]=="B1"}
    for x in validations:
        ref=baseline.get((x["stage"],x["seed"],x["step"]))
        if ref is None:continue
        losses=[];rank=[]
        for a,b in zip(ref["probe_trees"],x["probe_trees"]):
            aa=[json.dumps(t,separators=(",",":")) for t in a];bb=[json.dumps(t,separators=(",",":")) for t in b]
            if not aa and not bb:continue
            losses.append(1-2*len(set(aa)&set(bb))/max(len(aa)+len(bb),1))
            rank.extend(int((aa[i] if i<len(aa) else None)!=(bb[i] if i<len(bb) else None)) for i in range(max(len(aa),len(bb))))
        x["candidate_set_disagreement_vs_B1"]=float(np.mean(losses)) if losses else 0.
        x["candidate_rank_disagreement_vs_B1"]=float(np.mean(rank)) if rank else 0.
    atomic_json(validations,root.parent/"stability_validation.json")
    pd.DataFrame([{k:v for k,v in x.items() if k!="probe_trees"} for x in validations]).to_csv(root.parent/"stability_validation.csv",index=False)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output-dir",default="reports/cky_v25_precision/stability_runs")
    p.add_argument("--config",default="configs/precision/stability_smoke.yaml");p.add_argument("--cache-dir",default="cache/tinystories_integration")
    p.add_argument("--worker",action="store_true");p.add_argument("--arm",default="B2");p.add_argument("--seed",type=int,default=42)
    p.add_argument("--steps",type=int,default=100);p.add_argument("--stage",default="screen");p.add_argument("--resume-from")
    p.add_argument("--effective-tokens",type=int,default=16);p.add_argument("--eval-every",type=int,default=100)
    p.add_argument("--resume",action="store_true");p.add_argument("--screen-only",action="store_true")
    p.add_argument("--long-arms",nargs="+");args=p.parse_args()
    if args.worker:return worker(args)
    root=Path(args.output_dir);(root/"logs").mkdir(parents=True,exist_ok=True)
    def run(arm,seed,steps,stage):
        target=root/stage/f"{arm}_seed{seed}"
        if args.resume and (target/"state.json").exists() and json.loads((target/"state.json").read_text(encoding="utf-8"))["step"]>=steps:return
        cmd=[sys.executable,__file__,"--worker","--arm",arm,"--seed",str(seed),"--steps",str(steps),"--stage",stage,
             "--output-dir",str(root),"--config",args.config,"--cache-dir",args.cache_dir,"--effective-tokens",str(args.effective_tokens),"--eval-every",str(args.eval_every)]
        resume=target/"checkpoint.pt"
        if not resume.exists() and stage=="long" and seed==42:resume=root/"screen"/f"{arm}_seed42"/"checkpoint.pt"
        if resume.exists() and args.resume:cmd.extend(["--resume-from",str(resume)])
        print("Stability",stage,arm,seed,steps,flush=True)
        with (root/"logs"/f"{stage}_{arm}_{seed}.log").open("a" if args.resume else "w",encoding="utf-8") as f:
            result=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError(f"Stability {stage}/{arm}/{seed} failed; see log")
        aggregate(root)
    for arm in ["A1","A2","B1","B2","C2","D2"]:run(arm,42,100,"screen")
    if args.screen_only:return
    selected=args.long_arms
    if selected is None:
        # Predeclared gate: finite/no persistent skips, validation within .1 NLL
        # of best lazy; prefer lower fixed-weight paired gradient L2, then NLL.
        candidates=[]
        for arm in ["A2","B2","C2","D2"]:
            val=json.loads((root/"screen"/f"{arm}_seed42"/"validation.json").read_text(encoding="utf-8"))[-1]
            state=json.loads((root/"screen"/f"{arm}_seed42"/"state.json").read_text(encoding="utf-8"))
            numerical=json.loads((root.parent/f"numerical_{arm[0]}.json").read_text(encoding="utf-8"))
            main=next(x for x in numerical["models"] if x["config"]=="main_125m")
            gradient_error=max(x.get("normalized_l2_error",0) for x in main["gradients"].values())
            root_error=main["forward"]["semantic"]["normalized_l2_error"]
            if state["skipped_steps"]<=5 and gradient_error<=.05:candidates.append((arm,root_error,val["val_nll"],gradient_error))
        best=min(x[2] for x in candidates);eligible=[x for x in candidates if x[2]<=best+.1]
        selected=["B1"]+[x[0] for x in sorted(eligible,key=lambda x:(x[1],x[2]))[:2]]
        atomic_json(dict(rule="B1 reference + two finite lazy policies within .1 NLL of best, lowest paired root semantic L2 first (full-model gradient L2 <=.05 gate)",candidates=candidates,selected=selected),root.parent/"selection.json")
    for seed in [13,42,2026]:
        for arm in selected:run(arm,seed,500,"long")

if __name__=="__main__":main()
