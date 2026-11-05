"""Standalone source-probe replay of native Legacy search, never used in training/timing.

The generated diagnostic function is byte-identical to the production function
when the injected _probe lines are removed. No scores or native ops are replaced.
"""
from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))  # repo root: run from the repo root or via python -m
del _sys, _Path
from contextlib import contextmanager
import inspect,textwrap,types,gc,argparse
from pathlib import Path
from collections import Counter
import pandas as pd
import torch
from models.forest_lazy import LazyForestMixin
from models.model import LatentForestLM
from models.precision_policy import POLICIES,configure_model
from legacy.cky_v2.benchmark_cky_v2 import CASES,cpu_snapshot,snapshot_hash
from profile_performance import make_batch
from train import make_optimizer
from legacy.precision_v25.benchmark_precision_v25 import restore,step
from utils import atomic_json,load_config,rng_state,restore_rng,set_seed,to_device
from legacy.cky_v3.v3_study import assert_rng_equal

class WidthProbe:
    def __init__(self,profile=False):self.profile=profile;self.starts={};self.events=[];self.counts={}
    def begin(self,width,name):
        if self.profile:
            ctx=torch.profiler.record_function(f'V3Baseline/width_{width}/{name}');ctx.__enter__();self.starts[width,name]=ctx
        else:
            a=torch.cuda.Event(enable_timing=True);a.record();self.starts[width,name]=a
    def end(self,width,name):
        start=self.starts.pop((width,name))
        if self.profile:start.__exit__(None,None,None)
        else:
            b=torch.cuda.Event(enable_timing=True);b.record();self.events.append((width,name,start,b))
    def capture(self,width,valid):
        if not self.profile:self.counts[width]=(valid.numel(),valid.sum())
    def rows(self):
        rows={w:dict(width=w,candidates=c,valid_candidates=int(v.cpu())) for w,(c,v) in self.counts.items()}
        for w,name,a,b in self.events:rows[w][name]=a.elapsed_time(b)
        return list(rows.values())

@contextmanager
def legacy_source_probe(forest,probe):
    original=textwrap.dedent(inspect.getsource(LazyForestMixin._search_scores));source=original
    injections=[
        ('        starts = list(range(w - width + 1))','        _probe.begin(width,"elapsed_ms"); _probe.begin(width,"gather_pair_ms")'),
        ('        e = score.shape[-1]','        _probe.end(width,"gather_pair_ms"); _probe.begin(width,"selection_ms")'),
        ('        retained_flat = chosen.gather(2, rank)','        _probe.end(width,"selection_ms"); _probe.begin(width,"decode_ms")'),
        ('        lp, rp = decoded[...,0]+rows*decoded[...,2], decoded[...,1]+rows*decoded[...,3]','        _probe.end(width,"decode_ms"); _probe.begin(width,"assembly_ms")'),
        ('        total_spans = total_spans + (lengths[:, None] >= torch.tensor([i + width for i in starts], device=lengths.device)).sum()',
         '        _probe.end(width,"assembly_ms"); _probe.end(width,"elapsed_ms"); _probe.capture(width,valid)'),
    ]
    for marker,extra in injections:
        assert source.count(marker)==1,marker;source=source.replace(marker,marker+'\n'+extra)
    clean=''.join(line for line in source.splitlines(keepends=True) if not line.lstrip().startswith('_probe.'))
    assert clean==original,'Instrumentation changed original numerical source'
    namespace=dict(LazyForestMixin._search_scores.__globals__);namespace['_probe']=probe
    exec(compile(source,'<read-only legacy width probes>','exec'),namespace)
    forest._search_scores=types.MethodType(namespace['_search_scores'],forest)
    try:yield
    finally:del forest._search_scores


def kernel_widths(prof):
    counts=Counter()
    for event in prof.events():
        if event.device_type!=torch.autograd.DeviceType.CPU or not event.name.startswith('cudaLaunchKernel'):continue
        parent=event.cpu_parent
        while parent is not None:
            if parent.name.startswith(('V3Baseline/width_','V3/width_')) and parent.name.endswith('/elapsed_ms'):
                counts[int(parent.name.split('/')[1].split('_')[1])]+=1;break
            parent=parent.cpu_parent
    return counts


def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',default='reports/cky_v3');p.add_argument('--cache-dir',default='cache/tinystories_integration');a=p.parse_args()
    root=Path(a.output_dir);formal=pd.read_csv(root/'benchmark.csv');records=[];audits=[];memory=[]
    for precision,name in formal[['precision','case']].drop_duplicates().itertuples(index=False,name=None):
        set_seed(42);torch.set_num_threads(4);pp=POLICIES['B' if precision=='bf16' else 'E'];pp.configure_backend()
        _,config,size,words,real=next(c for c in CASES if c[0]==name);cfg=load_config(f'configs/{config}.yaml');cfg['data']['cache_dir']=a.cache_dir
        model=configure_model(LatentForestLM(cfg['model']).cuda().train(),pp);opt=make_optimizer(model,cfg['train'])
        batch=to_device(make_batch(cfg,size,4,words,128,'cache' if real else 'synthetic'),torch.device('cuda'))
        for param in model.parameters():param.grad=torch.zeros_like(param)
        opt.step();opt.zero_grad(set_to_none=True);sha=snapshot_hash(cpu_snapshot(model.state_dict()));assert formal[(formal.case==name)&(formal.precision==precision)].weight_sha256.eq(sha).all()
        initial=rng_state();state=dict(model=cpu_snapshot(model.state_dict()),optimizer=cpu_snapshot(opt.state_dict()),scaler={},rng=initial)
        scaler=pp.make_scaler();state['scaler']=scaler.state_dict();outputs=[];ends=[]
        for impl in ('lazy_semantic','lazy_tensorized'):
            model.forest.implementation=impl
            if impl=='lazy_semantic':
                if hasattr(model.forest,'_geometry_cache'):model.forest._geometry_cache.clear()
            else:model.forest._lookup_cache.clear()
            for _ in range(2):
                with pp.outer_context('cuda'):out=model(**batch)
                del out
            restore(model,opt,scaler,state,impl);torch.cuda.reset_peak_memory_stats()
            assert not step(model,batch,opt,pp,scaler);torch.cuda.synchronize()
            memory.append(dict(case=name,precision=precision,implementation=impl,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                inactive_implementation_cache_cleared=True,weight_sha256=sha,scope='Separate full-update memory audit after warmup2, only active implementation static cache resident'))
            restore(model,opt,scaler,state,impl)
            model.forest.collect_timing=True
            for repeat in range(3):
                restore_rng(initial);probe=WidthProbe()
                if impl=='lazy_semantic':
                    with legacy_source_probe(model.forest,probe),pp.outer_context('cuda'):out=model(**batch)
                    rows=probe.rows()
                else:
                    with pp.outer_context('cuda'):out=model(**batch)
                    rows=model.forest.last_workload['width_profile']
                records.extend(dict(case=name,precision=precision,implementation=impl,repeat=repeat,**row) for row in rows)
                if repeat==0:outputs.append((out['logits'].detach().cpu(),out['memory']['scores'].detach().cpu()));ends.append(rng_state())
                del out
            model.forest.collect_timing=False;model.forest.v3_profile=True;restore_rng(initial)
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
                if impl=='lazy_semantic':
                    with legacy_source_probe(model.forest,WidthProbe(True)),pp.outer_context('cuda'):out=model(**batch)
                else:
                    with pp.outer_context('cuda'):out=model(**batch)
                torch.cuda.synchronize()
            counts=kernel_widths(prof);del out;model.forest.v3_profile=False
            for row in records:
                if row['case']==name and row['precision']==precision and row['implementation']==impl:row['kernel_launches']=counts[row['width']]
        assert_rng_equal(*ends);assert all(torch.equal(x,y) for x,y in zip(*outputs))
        audits.append(dict(case=name,precision=precision,weight_sha256=sha,native_outputs_bitwise_equal=True,rng_equal=True,source_probe_only=True))
        pd.DataFrame(records).to_csv(root/'width_diagnostics.csv',index=False);atomic_json(audits,root/'width_diagnostics_audit.json');atomic_json(memory,root/'memory_profile.json')
        del model,opt,batch,outputs,state;gc.collect();torch.cuda.empty_cache();print(name,precision,'width diagnostics complete',flush=True)

if __name__=='__main__':main()
