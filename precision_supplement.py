"""Independent score-accumulation and FP16 scaling controls."""
import argparse,json
from pathlib import Path
import torch
from models.composer import Composer
from models.forest import SemanticForest
from models.model import LatentForestLM
from models.precision_policy import POLICIES,configure_model
from precision_study import errors,tensor_health,margins_observer
from validate_cky_v25 import model_batch
from benchmark_cky_v2 import cpu_snapshot
from utils import set_seed,load_config,rng_state,restore_rng,atomic_json,atomic_torch_save


def scaling(group,root):
    policy=POLICIES[group];policy.configure_backend();torch.set_num_threads(1)
    source=torch.load(root/"fixed_inputs.pt",map_location="cpu",weights_only=False);cfg=source["config"]
    forest=SemanticForest(Composer(cfg["semantic_size"],cfg["composer_hidden"]),4,12,cfg["prune_delta"],cfg["exploration"],"lazy_semantic").cuda().train()
    forest.precision_policy=policy;forest.composer.load_state_dict(source["composer"])
    results=[];snapshots=[]
    for enabled in ([True,False] if group=="D" else [False]):
        xs=[source[k].cuda().requires_grad_() for k in ("leaves","scores","coeff")];mask=source["mask"].cuda()
        forest.zero_grad(set_to_none=True);restore_rng(source["rng"])
        with policy.outer_context("cuda"):
            out=forest(*xs,mask)
            probe=torch.linspace(-.5,.5,out["semantic"].numel(),device="cuda").reshape_as(out["semantic"])
            loss=(out["semantic"]*probe).mean()+out["log_prior"].square().mean()*.001
        scale=policy.scaler_init_scale if enabled else 1.;(loss*scale).backward()
        grads={n:p.grad.detach().cpu()/scale for n,p in forest.composer.named_parameters() if p.grad is not None}
        snapshots.append(grads)
        results.append(dict(scaling=enabled,scale=scale,loss=float(loss.detach()),
            parameter_gradients={n:tensor_health(g) for n,g in grads.items()},leaf_gradient=tensor_health(xs[0].grad/scale)))
    atomic_torch_save(snapshots,root/f"underflow_grads_{group}.pt")
    atomic_json(dict(group=group,results=results,note="Isolated identical root-probe; scaling-on/off are diagnostics, not additional training arms"),root/f"underflow_{group}.json")


def score(root):
    policy=POLICIES["B"];policy.configure_backend();torch.set_num_threads(1);set_seed(42)
    cfg=load_config("configs/main_125m.yaml");m=configure_model(LatentForestLM(cfg["model"]).cuda().train(),policy)
    batch=model_batch(cfg,batch=1,words=24,histories=4,target_length=16,device="cuda");ids=batch["history_ids"].reshape(-1,24);mask=batch["word_mask"].reshape(-1,24)
    leaves=m.semantic_embedding(ids)*mask.unsqueeze(-1)
    with policy.outer_context("cuda"):scores,u=m.syntax(ids,mask)
    m.forest.implementation="lazy_semantic"
    initial=rng_state();outputs=[];grads=[];margins=[]
    for cast in (False,True):
        m.zero_grad(set_to_none=True);restore_rng(initial);rows=[];m.forest.precision_observer=margins_observer(rows)
        with policy.outer_context("cuda"):
            out=m.forest(leaves,scores.float() if cast else scores,u,mask,return_spans=True)
            probe=torch.linspace(-.5,.5,out["semantic"].numel(),device="cuda").reshape_as(out["semantic"])
            loss=(out["semantic"]*probe).mean()+out["log_prior"].square().mean()*.001
        loss.backward(retain_graph=not cast)
        outputs.append(cpu_snapshot(out));grads.append({n:p.grad.detach().cpu().clone() for n,p in m.syntax.span_scorer.named_parameters() if p.grad is not None});margins.append(rows)
    changed_ids=changed_masks=0;total=0
    for key,a in outputs[0]["spans"].items():
        b=outputs[1]["spans"][key]
        for label in ("selected_flat","retained_flat"):
            if a[label] is not None:changed_ids+=int(a[label].ne(b[label]).sum());total+=a[label].numel()
        changed_masks+=int(a["candidate_mask"].ne(b["candidate_mask"]).sum())
    from validate_cky_v25 import gradient_statistics
    result=dict(policy=policy.state_dict(),original_local_score_dtype=str(scores.dtype),
        existing_leaf_dtype=str(outputs[0]["spans"][0,1]["scores"].dtype),fp32_leaf_dtype=str(outputs[1]["spans"][0,1]["scores"].dtype),
        existing_internal_dtype=str(outputs[0]["spans"][0,2]["scores"].dtype),
        root_score=errors(outputs[0]["scores"],outputs[1]["scores"]),log_prior=errors(outputs[0]["log_prior"],outputs[1]["log_prior"]),
        changed_ids=changed_ids,id_entries=total,changed_masks=changed_masks,span_scorer_gradient=gradient_statistics(*grads),margins=margins,
        note="Same BF16 local potentials; only cast before CKY. Does not recover scalar-head quantization. Scores and Composer were not changed together.")
    atomic_json(result,root/"score_precision_control.json")


def main():
    p=argparse.ArgumentParser();p.add_argument("--mode",choices=["score","D","E","compare"],required=True)
    p.add_argument("--output-dir",default="reports/cky_v25_precision");a=p.parse_args();root=Path(a.output_dir)
    if a.mode=="score":score(root)
    elif a.mode in ("D","E"):scaling(a.mode,root)
    else:
        e=torch.load(root/"underflow_grads_E.pt",weights_only=False)[0];d=torch.load(root/"underflow_grads_D.pt",weights_only=False)
        rows=[]
        for q,g in enumerate(d):
            rows.append(dict(scaled=q==0,zero_where_E_nonzero=sum(int((e[n].ne(0)&g[n].eq(0)).sum()) for n in e),
                             error={n:errors(e[n],g[n]) for n in e}))
        atomic_json(dict(results=rows,note="Zero-gradient mismatch is an underflow indicator, not proof: analytic derivatives also depend on arithmetic precision"),root/"underflow_comparison.json")

if __name__=="__main__":main()
