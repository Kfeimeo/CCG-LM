"""Repeatability and first-divergence controls; not changes to training policy."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))  # repo root: run from the repo root or via python -m
del _sys, _Path
import argparse,json
from pathlib import Path
import torch
from torch.nn.attention import sdpa_kernel,SDPBackend
from contextlib import nullcontext
from models.model import LatentForestLM
from models.precision_policy import POLICIES,configure_model
from legacy.cky_v25.validate_cky_v25 import model_batch
from legacy.precision_v25.precision_study import errors
from utils import set_seed,load_config,rng_state,restore_rng,atomic_json


def run(group,root):
    policy=POLICIES[group];policy.configure_backend();torch.set_num_threads(1);set_seed(42)
    cfg=load_config("configs/main_125m.yaml");m=configure_model(LatentForestLM(cfg["model"]).cuda(),policy)
    batch=model_batch(cfg,batch=1,words=24,histories=4,target_length=16,device="cuda");initial=rng_state();cases=[]
    for training,math_only,label in [(True,False,"native_train"),(False,False,"native_eval"),(True,True,"math_train")]:
        m.train(training);captures=[];rngs=[]
        for impl in ("score_first","score_first","lazy_semantic"):
            values={};states={};handles=[]
            for name,module in m.named_modules():
                if name in ("decoder.dropout","decoder.norm") or (name.startswith("decoder.blocks.") and name.count(".")==2) or (name.startswith("readers.") and name.count(".")<=2):
                    def pre(mod,inp,name=name):
                        states[name]=torch.cuda.get_rng_state().clone()
                        if name.startswith("readers."):values[name+"_input"]=inp[0].detach().cpu().clone()
                    def post(mod,inp,out,name=name):
                        values[name]=((out[0] if isinstance(out,tuple) else out).detach().cpu().clone())
                    handles.append(module.register_forward_pre_hook(pre));handles.append(module.register_forward_hook(post))
            m.forest.implementation=impl;restore_rng(initial)
            context=sdpa_kernel(SDPBackend.MATH) if math_only else nullcontext()
            with context,policy.outer_context("cuda"):out=m(**batch)
            values["logits"]=out["logits"].detach().cpu();values["root"]=out["memory"]["semantic"].detach().cpu()
            captures.append(values);rngs.append(states)
            for h in handles:h.remove()
            del out
        for q,comparison in [(1,"same_impl_repeat"),(2,"paired_impl")]:
            metrics={k:errors(captures[0][k],captures[q][k]) for k in captures[0]}
            cases.append(dict(condition=label,comparison=comparison,metrics=metrics,
                module_entry_rng_equal={k:torch.equal(rngs[0][k],rngs[q][k]) for k in rngs[0]}))
    result=dict(group=group,cases=cases,note="Evaluation and math-only SDPA are diagnostic controls; they are not the benchmark/training configuration")
    atomic_json(result,root/f"controls_{group}.json")
    for c in cases:print(group,c["condition"],c["comparison"],c["metrics"]["logits"]["normalized_l2_error"],flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--group",choices=list(POLICIES),default="B");p.add_argument("--output-dir",default="reports/cky_v25_precision")
    a=p.parse_args();run(a.group,Path(a.output_dir))
