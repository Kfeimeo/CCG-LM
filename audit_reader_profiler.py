"""Separate one-sample profiled update timing; never used for speedup ratios."""
from __future__ import annotations
import argparse,gc,json,subprocess,sys,time
from pathlib import Path
import torch
from benchmark_reader_precision import ARMS,SELECTED
from benchmark_cky_v2 import CASES,cpu_snapshot,snapshot_hash
from benchmark_precision_v25 import restore,step
from models.model import LatentForestLM
from models.precision_policy import configure_model,backend_state
from reader_precision_study import READER_POLICIES,set_reader_arm
from profile_performance import make_batch
from train import make_optimizer
from utils import atomic_json,load_config,rng_state,set_seed,to_device


def run(case,args):
    name,config,size,words,real=case;torch.set_num_threads(4);set_seed(42)
    READER_POLICIES['B'].configure_backend()
    cfg=load_config(f'configs/{config}.yaml');cfg['data']['cache_dir']=args.cache_dir
    model=configure_model(LatentForestLM(cfg['model']).cuda().train(),READER_POLICIES['B'])
    opt=make_optimizer(model,cfg['train']);scaler=READER_POLICIES['B'].make_scaler()
    batch=to_device(make_batch(cfg,size,4,words,128,'cache' if real else 'synthetic'),torch.device('cuda'))
    for p in model.parameters():p.grad=torch.zeros_like(p)
    opt.step();opt.zero_grad(set_to_none=True);torch.cuda.synchronize()
    state=dict(model=cpu_snapshot(model.state_dict()),optimizer=cpu_snapshot(opt.state_dict()),scaler=scaler.state_dict(),rng=rng_state())
    weight_hash=snapshot_hash(state['model']);root=Path(args.output_dir)
    import pandas as pd
    expected=pd.read_csv(root/'performance'/name/'benchmark.csv')
    assert expected.weight_sha256.eq(weight_hash).all()
    records=[]
    for label,family,impl in ARMS:
        restore(model,opt,scaler,state,impl);policy=set_reader_arm(model,family,impl)
        for _ in range(2):assert not step(model,batch,opt,policy,scaler)
        restore(model,opt,scaler,state,impl)
        timing={}
        def profiled_update(m,bb,o,unused):
            start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            tick=time.perf_counter();start.record();skipped=step(m,bb,o,policy,scaler);end.record();end.synchronize()
            timing.update(profiled_update_wall_ms=(time.perf_counter()-tick)*1000,
                          profiled_update_cuda_envelope_ms=start.elapsed_time(end))
            assert not skipped
        import benchmark_cky_v25 as old
        original=old.regular_step;old.regular_step=profiled_update
        try:counts=old.operator_counts(model,batch,opt,policy.outer_autocast)
        finally:old.regular_step=original
        records.append(dict(arm=label,**timing,**counts,backend=backend_state()))
    atomic_json(dict(case=name,weight_sha256=weight_hash,weight_matches_formal_benchmark=True,
        scope='Supplementary separate profiler replay, one sample/arm after two warmups; not the formal no-profiler round; no speedup claims',
        records=records),root/'performance'/name/'profiler_timing.json')
    print(name,'profile timing archived',flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',default='reports/cky_v25_reader')
    p.add_argument('--cache-dir',default='cache/tinystories_integration');p.add_argument('--worker',action='store_true');p.add_argument('--case',choices=sorted(SELECTED))
    a=p.parse_args();root=Path(a.output_dir)
    if a.worker:return run(next(c for c in CASES if c[0]==a.case),a)
    for case in CASES:
        if case[0] not in SELECTED:continue
        cmd=[sys.executable,__file__,'--worker','--case',case[0],'--output-dir',str(root),'--cache-dir',a.cache_dir]
        with (root/'logs'/f'profiler_timing_{case[0]}.log').open('w',encoding='utf-8') as f:
            r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
        if r.returncode:raise RuntimeError('Profiler timing replay failed: '+case[0])
        print(case[0],'profiler replay done',flush=True)

if __name__=='__main__':main()
