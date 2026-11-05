"""Fixed-weight main-model Reader isolation; no training or tolerance changes."""
from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))  # repo root: run from the repo root or via python -m
del _sys, _Path
import argparse,gc,json,subprocess,sys
from pathlib import Path
import torch
from models.model import LatentForestLM
from models.precision_policy import configure_model,backend_state
from legacy.reader_v25.reader_precision_study import READER_POLICIES,ReaderObserver,tensor_errors,global_gradients,set_reader_arm
from legacy.cky_v25.validate_cky_v25 import model_batch,compare_forests,gradient_statistics
from legacy.cky_v2.benchmark_cky_v2 import cpu_snapshot,snapshot_hash
from legacy.precision_v25.precision_study import search_equal
from utils import atomic_json,set_seed,load_config,rng_state,restore_rng,device_info


def attach(model,save=True,attention=True,cache=None):
    observers={};entry={};exit_rng={};blocks={};handles=[]
    for name,reader in model.readers.items():
        obs=ReaderObserver(save,attention);reader.precision_observer=obs;observers[name]=obs
        def before(module,inp,name=name):
            entry[name]=torch.cuda.get_rng_state().clone()
            if cache is not None:
                hidden,mem,*rest=inp;work=dict(mem)
                for key in ("semantic","vector_coeff","log_prior","candidate_mask"):
                    work[key]=cache[key].to(hidden.device)
                return (hidden,work,*rest)
        def after(module,inp,out,name=name,obs=obs):
            obs.record("return_to_decoder",out[0]);exit_rng[name]=torch.cuda.get_rng_state().clone()
        handles.extend([reader.register_forward_pre_hook(before),reader.register_forward_hook(after)])
        next_index=int(name)
        if next_index<len(model.decoder.blocks):
            handles.append(model.decoder.blocks[next_index].register_forward_hook(lambda m,i,o,name=name:blocks.__setitem__(name,o.detach().cpu().clone())))
    def remove():
        for h in handles:h.remove()
        for r in model.readers.values():r.precision_observer=None
    return observers,entry,exit_rng,blocks,remove


def compare_reader_values(a,b):
    rows={}
    for layer in a:
        assert a[layer].values.keys()==b[layer].values.keys()
        rows[layer]={k:tensor_errors(a[layer].values[k],b[layer].values[k]) for k in a[layer].values}
    return rows


def run_case(batch_size,args):
    torch.set_num_threads(1);READER_POLICIES["B"].configure_backend();set_seed(42)
    cfg=load_config("configs/main_125m.yaml")
    model=configure_model(LatentForestLM(cfg["model"]).cuda().train(),READER_POLICIES["B"])
    model.forest.capture_spans=True
    weight_hash=snapshot_hash(cpu_snapshot(model.state_dict()))
    batch=model_batch(cfg,batch=batch_size,words=24,histories=4,target_length=args.target_length,device="cuda")
    input_hash=snapshot_hash({k:v.detach().cpu() for k,v in batch.items()})
    initial=rng_state();result={};control={};dtype_rows={};root_cache={}
    families=args.families
    for family in families:
        policy=READER_POLICIES[family];outs=[];gradients=[];traces=[];entries=[];exits=[];next_blocks=[];end_states=[]
        for impl in ("score_first","lazy_semantic"):
            set_reader_arm(model,family,impl);model.zero_grad(set_to_none=True);restore_rng(initial)
            observers,en,ex,nb,remove=attach(model)
            with policy.outer_context("cuda"):out=model(**batch)
            end_states.append(rng_state())
            # These are read-only CPU observations; the LM backward uses its real graph.
            cpu=dict(memory=cpu_snapshot(out["memory"]),logits=out["logits"].detach().cpu(),loss=out["loss"].detach().cpu())
            out["loss"].backward()
            gradients.append({n:None if p.grad is None else p.grad.detach().cpu().clone() for n,p in model.named_parameters()})
            outs.append(cpu);traces.append(observers);entries.append(en);exits.append(ex);next_blocks.append(nb)
            remove();del out
        structure=search_equal(outs[0]["memory"],outs[1]["memory"]);assert structure["identical"]
        compare_forests(outs[0]["memory"],outs[1]["memory"],family!="E")
        assert torch.equal(end_states[0]["torch"],end_states[1]["torch"])
        assert all(torch.equal(x,y) for x,y in zip(end_states[0]["cuda"],end_states[1]["cuda"]))
        forward=dict(root={k:tensor_errors(outs[0]["memory"][k],outs[1]["memory"][k]) for k in ("semantic","scores","log_prior","vector_coeff")},
            readers=compare_reader_values(*traces),next_decoder_blocks={k:tensor_errors(next_blocks[0][k],next_blocks[1][k]) for k in next_blocks[0]},
            lm_logits=tensor_errors(outs[0]["logits"],outs[1]["logits"]),loss=tensor_errors(outs[0]["loss"],outs[1]["loss"]))
        grad=global_gradients(*gradients,strict_label="fp32" if family=="E" else "bf16")
        # Per-parameter norms and errors preserve small/null directions explicitly.
        grad["parameters"]={n:gradient_statistics({n:a},{n:gradients[1][n]}) for n,a in gradients[0].items()}
        result[family]=dict(policy=policy.state_dict(),structure=structure,forward=forward,gradient=grad,
            entry_rng_equal={k:torch.equal(entries[0][k],entries[1][k]) for k in entries[0]},
            exit_rng_equal={k:torch.equal(exits[0][k],exits[1][k]) for k in exits[0]},end_rng_equal=True)
        dtype_rows[family]=[{k:o.summary() for k,o in arm.items()} for arm in traces]
        root_cache[family]=[{k:arm["memory"][k] for k in ("semantic","vector_coeff","log_prior","candidate_mask")} for arm in outs]
        reference_memory=root_cache[family][0]
        # Controlled input: explicit no-grad forward, model.train and Dropout remain ON.
        controls=[];control_entries=[];normal_no_grad=[]
        for identical in (False,True):
            cv=[];ce=[];cl=[];cnext=[];cw=[]
            for impl in ("score_first","lazy_semantic"):
                set_reader_arm(model,family,impl);restore_rng(initial)
                ob,en,ex,nb,remove=attach(model,attention=False,cache=reference_memory if identical else None)
                with torch.no_grad(),policy.outer_context("cuda"):out=model(**batch)
                cv.append(ob);ce.append(en);cl.append(out["logits"].detach().cpu());cnext.append(nb)
                cw.append({k:o.summary() for k,o in ob.items()});remove();del out
            control_case=dict(identical_reader_memory=identical,no_grad=True,dropout_training=True,backward_claim=False,
                readers=compare_reader_values(*cv),logits=tensor_errors(*cl),
                next_decoder_blocks={k:tensor_errors(cnext[0][k],cnext[1][k]) for k in cnext[0]},
                reader_entry_rng_equal={k:torch.equal(ce[0][k],ce[1][k]) for k in ce[0]},
                sdpa=[{k:v["sdpa"] for k,v in arm.items()} for arm in cw])
            controls.append(control_case)
        control[family]=controls
        print("main B",batch_size,family,"grad L2",grad["global_statistics"]["normalized_l2_gradient_error"],flush=True)
        del gradients,outs,traces;gc.collect();torch.cuda.empty_cache()
    comparisons={}
    if "B" in result and "R" in result:
        for q,impl in enumerate(("score_first","lazy_semantic")):
            comparisons[impl]={k:tensor_errors(root_cache["B"][q][k],root_cache["R"][q][k]) for k in ("semantic","vector_coeff","log_prior")}
            assert all(x["max_absolute_error"]==0 for x in comparisons[impl].values()),"Reader precision changed its upstream roots"
        b=result["B"]["gradient"]["global_statistics"]["normalized_l2_gradient_error"]
        r=result["R"]["gradient"]["global_statistics"]["normalized_l2_gradient_error"]
        comparisons["improvement_ratio"]=b/r if r>1e-30 else None
        comparisons["denominator_near_zero"]=r<1e-8
    root=Path(args.output_dir);root.mkdir(parents=True,exist_ok=True)
    output=dict(batch=batch_size,words=24,target_length=args.target_length,model="main_125m",parameter_count=sum(p.numel() for p in model.parameters()),
        weight_sha256=weight_hash,input_sha256=input_hash,environment=device_info(),backend=backend_state(),
        cases=result,controlled_input=control,root_invariance=comparisons)
    atomic_json(output,root/f"case_b{batch_size}.json");atomic_json(dtype_rows,root/f"dtype_b{batch_size}.json")
    del model;gc.collect();torch.cuda.empty_cache()


def aggregate(root):
    cases=[json.loads(p.read_text(encoding="utf-8")) for p in sorted(root.glob("case_b*.json"))]
    atomic_json([{k:v for k,v in c.items() if k!="controlled_input"} for c in cases],root/"forward_comparison.json")
    atomic_json([dict(batch=c["batch"],families={g:r["gradient"] for g,r in c["cases"].items()},
                      root_invariance=c["root_invariance"]) for c in cases],root/"gradient_comparison.json")
    atomic_json([dict(batch=c["batch"],cases=c["controlled_input"]) for c in cases],root/"controlled_input.json")
    atomic_json({p.stem:json.loads(p.read_text(encoding="utf-8")) for p in root.glob("dtype_b*.json")},root/"dtype_trace.json")


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output-dir",default="reports/cky_v25_reader")
    p.add_argument("--batches",nargs="+",type=int,default=[1,4]);p.add_argument("--families",nargs="+",default=["B","R","E"],choices=list(READER_POLICIES))
    p.add_argument("--target-length",type=int,default=16);p.add_argument("--worker",action="store_true");p.add_argument("--batch",type=int);p.add_argument("--resume",action="store_true")
    args=p.parse_args();root=Path(args.output_dir);(root/"logs").mkdir(parents=True,exist_ok=True)
    if args.worker:return run_case(args.batch,args)
    for b in args.batches:
        if args.resume and (root/f"case_b{b}.json").exists():continue
        cmd=[sys.executable,__file__,"--worker","--batch",str(b),"--output-dir",str(root),"--target-length",str(args.target_length),"--families",*args.families]
        print("Reader numerical main batch",b,flush=True)
        with (root/"logs"/f"validate_b{b}.log").open("w",encoding="utf-8") as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
        if r.returncode:raise RuntimeError(f"Batch {b} failed; see log")
        aggregate(root)

if __name__=="__main__":main()
