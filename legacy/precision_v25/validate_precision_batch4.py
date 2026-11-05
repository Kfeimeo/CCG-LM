"""Additional full-model B4/W24 backward pairs, both coefficient modes."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))  # repo root: run from the repo root or via python -m
del _sys, _Path
import argparse
from pathlib import Path
import torch
from models.precision_policy import POLICIES
from legacy.precision_v25.validate_precision_v25 import full_model_case
from utils import atomic_json

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--group",choices=list("ABCDE"),required=True)
    p.add_argument("--output-dir",default="reports/cky_v25_precision");a=p.parse_args()
    torch.set_num_threads(1);POLICIES[a.group].configure_backend();root=Path(a.output_dir)
    results=[full_model_case(a.group,"smoke",mode,root,batch_override=4,words_override=24)
             for mode in ("forest_scalar4","forest_vector4")]
    atomic_json(dict(group=a.group,cases=results),root/f"backward_batch4_{a.group}.json")
    print(a.group,"B4/W24 scalar+vector backward complete",flush=True)
