"""Replay grad-enabled forwards solely to archive discrete search and RNG traces."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))  # repo root: run from the repo root or via python -m
del _sys, _Path
import argparse,hashlib,json,gc
from pathlib import Path
import torch
from models.model import LatentForestLM
from models.precision_policy import configure_model
from legacy.reader_v25.reader_precision_study import READER_POLICIES,set_reader_arm
from legacy.cky_v25.validate_cky_v25 import model_batch
from legacy.cky_v2.benchmark_cky_v2 import cpu_snapshot,snapshot_hash
from utils import set_seed,load_config,rng_state,restore_rng,atomic_json,atomic_torch_save


def encode(memory,word_mask):
    spans=memory["spans"];n=word_mask.shape[0]*word_mask.shape[1]
    bases={};offset=0;records=[]
    for (i,j),c in sorted(spans.items(),key=lambda x:(x[0][1]-x[0][0],x[0][0])):
        bases[i,j]=offset;offset+=n*c["scores"].shape[-1]
    for (i,j),c in sorted(spans.items(),key=lambda x:(x[0][1]-x[0][0],x[0][0])):
        k=c["scores"].shape[-1]
        row=dict(span=[i,j],candidate_base=bases[i,j],slots=k,mask=c["candidate_mask"].tolist(),
                 scores=c["scores"].float().tolist(),metadata=c["metadata"],
                 selected_flat=None if c["selected_flat"] is None else c["selected_flat"].tolist(),
                 retained_flat=None if c["retained_flat"] is None else c["retained_flat"].tolist())
        if j-i>1:
            left=[];right=[]
            for r in range(n):
                ll=[];rr=[]
                for s in range(k):
                    off,l,t=c["metadata"][int(c["retained_flat"][r,s])]
                    ll.append(bases[i,i+off]+r*spans[i,i+off]["scores"].shape[-1]+l)
                    rr.append(bases[i+off,j]+r*spans[i+off,j]["scores"].shape[-1]+t)
                left.append(ll);right.append(rr)
            row.update(left_id=left,right_id=right)
            if "left_id" in c:assert c["left_id"].tolist()==left and c["right_id"].tolist()==right
        records.append(row)
    lengths=word_mask.reshape(n,-1).sum(-1).tolist();mask=memory["candidate_mask"].reshape(n,-1)
    roots=[]
    for r,L in enumerate(lengths):
        k=spans[0,max(1,L)]["scores"].shape[-1]
        roots.append([bases[0,max(1,L)]+r*k+s if mask[r,s] else -1 for s in range(mask.shape[-1])])
    return dict(spans=records,root_ids=roots,candidate_id_definition="span base + row * actual slot count + retained slot")


def run(b,root):
    set_seed(42);torch.set_num_threads(1);READER_POLICIES["B"].configure_backend();cfg=load_config("configs/main_125m.yaml")
    m=configure_model(LatentForestLM(cfg["model"]).cuda().train(),READER_POLICIES["B"]);m.forest.capture_spans=True
    expected=json.loads((root/f"case_b{b}.json").read_text(encoding="utf-8"))
    assert snapshot_hash(cpu_snapshot(m.state_dict()))==expected["weight_sha256"]
    batch=model_batch(cfg,batch=b,words=24,histories=4,target_length=expected["target_length"],device="cuda");input_snapshot={k:v.cpu() for k,v in batch.items()};input_hash=snapshot_hash(input_snapshot)
    assert input_hash==expected["input_sha256"]
    initial=rng_state();all_rng={}
    for family in ("B","R","E"):
        pairs=[];rngs=[]
        for impl in ("score_first","lazy_semantic"):
            p=set_reader_arm(m,family,impl);restore_rng(initial);states={};handles=[]
            for name,reader in m.readers.items():
                handles.append(reader.register_forward_pre_hook(lambda mod,inp,name=name:states.__setitem__(name,torch.cuda.get_rng_state().clone())))
            with p.outer_context("cuda"):out=m(**batch)
            # Forward only, grad mode stays enabled so the native path matches backward cases.
            mem=cpu_snapshot(out["memory"]);pairs.append(encode(mem,batch["word_mask"].cpu()));rngs.append(states)
            for h in handles:h.remove()
            del out,mem
        assert pairs[0]==pairs[1]
        all_rng[family]=rngs
        atomic_json(dict(batch=b,family=family,weight_sha256=expected["weight_sha256"],
            input_sha256=input_hash,replay_forward_only=True,trace=pairs[0],entry_rng_sha256=[{k:hashlib.sha256(v.numpy().tobytes()).hexdigest() for k,v in st.items()} for st in rngs]),root/f"search_b{b}_{family}.json")
    atomic_torch_save(dict(initial=initial,reader_entry=all_rng,inputs=input_snapshot),root/f"rng_b{b}.pt")
    print("Search/RNG replay archived B",b,flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--batch",type=int,required=True);p.add_argument("--output-dir",default="reports/cky_v25_reader")
    a=p.parse_args();run(a.batch,Path(a.output_dir))
