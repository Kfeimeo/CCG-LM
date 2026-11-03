"""V3.5 Stage D: static index-cache build cost and capacity, measured in isolation.

(1) Build-time comparison, reference Python builder vs vectorized builder, for every
    (words, rows) key observed in the real probe stream (bitwise-equal plans asserted).
(2) Real-stream replay: consecutive real micro-batches with lazy_tensorized under
    LRU capacity 8 vs 32 and reference vs fast builder; wall time, misses, build time.
No search, gradient or data semantics change; the cache stores shape-only geometry.
"""
from __future__ import annotations
import argparse,gc,json,statistics,time
from pathlib import Path
import pandas as pd
import torch
from data.dataset import StoryDataset
from models.cky_index_cache import CKYIndexCache
from models.model import LatentForestLM
from models.precision_policy import PrecisionPolicy,configure_model
from profile_cky_v35 import probe_batches
from benchmark_cky_v35 import step
from train import make_optimizer
from utils import atomic_json,load_config,set_seed,to_device
from tests.test_index_cache_fast import same_plan


def build_times(args,root):
    keys=json.loads((Path(args.audit_dir)/'cache_analysis.json').read_text(encoding='utf-8'))['per_micro_batch']
    observed=sorted({(int(k['words']),int(k['rows'])) for m,v in keys.items() if m!='builds' for k in v['key_frequency']}|{(24,16),(24,64),(24,128),(24,512),(13,4),(18,32)})
    device=torch.device('cuda');rows=[]
    for words,n in observed:
        samples={'reference':[],'fast':[]}
        for builder in ('reference','fast'):
            fn=CKYIndexCache._build_reference if builder=='reference' else CKYIndexCache._build_fast
            if builder=='fast':CKYIndexCache._unit_geometry.clear()
            for rep in range(3):
                torch.cuda.synchronize();tick=time.perf_counter();plan=fn(words,n,4,12,device,True);torch.cuda.synchronize();samples[builder].append((time.perf_counter()-tick)*1000)
                if rep==0 and builder=='fast':same_plan(reference_plan,plan)
                if builder=='reference' and rep==0:reference_plan=plan
        rows.append(dict(words=words,rows=n,reference_ms_min=min(samples['reference']),reference_ms_median=statistics.median(samples['reference']),
            fast_cold_ms=samples['fast'][0],fast_warm_unit_ms_min=min(samples['fast'][1:]),fast_warm_unit_ms_median=statistics.median(samples['fast'][1:]),device_bytes=plan.device_bytes))
        del plan,reference_plan;torch.cuda.empty_cache()
    df=pd.DataFrame(rows);df.to_csv(root/'cache_build_times.csv',index=False);print(df.to_string(index=False),flush=True);return df


def stream_replay(args,root):
    device=torch.device('cuda');cfg=load_config('configs/v35/baseline.yaml');cfg['data']['cache_dir']=args.cache_dir
    policy=PrecisionPolicy(**cfg['precision']);policy.configure_backend();set_seed(42);torch.set_num_threads(4)
    data=StoryDataset(args.cache_dir);rows=[]
    for mbs in args.micro_batches:
        batches=probe_batches(data,mbs,4096,args.steps,skip_updates=3)
        shapes=[(int(b['history_ids'].shape[-1]),int(b['history_ids'].shape[0])*int(b['history_ids'].shape[1])) for b in batches]
        for capacity,builder in [(8,'reference'),(8,'fast'),(32,'fast'),(64,'fast')]:
            set_seed(42);model=configure_model(LatentForestLM(cfg['model'],42).to(device).train(),policy);opt=make_optimizer(model,cfg['train'])
            for p in model.parameters():p.grad=torch.zeros_like(p)
            opt.step();opt.zero_grad(set_to_none=True)
            CKYIndexCache.fast_build=builder=='fast';CKYIndexCache._unit_geometry.clear();model.forest._geometry_cache=CKYIndexCache(capacity)
            gpu=[to_device(b,device) for b in batches[:2]]
            for g in gpu:step(model,g,opt,policy)  # warm kernels only; cache cleared again below
            model.forest._geometry_cache=CKYIndexCache(capacity);CKYIndexCache._unit_geometry.clear();del gpu;gc.collect();torch.cuda.empty_cache()
            per=[];torch.cuda.synchronize();tick=time.perf_counter()
            for b in batches:
                g=to_device(b,device);torch.cuda.synchronize();t0=time.perf_counter();step(model,g,opt,policy);torch.cuda.synchronize();per.append((time.perf_counter()-t0)*1000);del g
            total=(time.perf_counter()-tick)*1000;st=model.forest._geometry_cache.statistics()
            row=dict(micro_batch_size=mbs,steps=len(batches),unique_keys=len(set(shapes)),capacity=capacity,builder=builder,total_wall_ms=total,median_step_ms=statistics.median(per),
                     misses=st['misses'],hits=st['hits'],hit_rate=st['hit_rate'],evictions=st['evictions'],build_ms_total=st['build_host_ms_total'],build_fraction=st['build_host_ms_total']/total,
                     resident_bytes=st['resident_bytes'],peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
            rows.append(row);print(json.dumps(row),flush=True)
            del model,opt;gc.collect();torch.cuda.empty_cache()
        CKYIndexCache.fast_build=True
    df=pd.DataFrame(rows);df.to_csv(root/'cache_stream_replay.csv',index=False);return df


def optimizer_check(args,root):
    """AdamW implementation micro-check (diagnostic only; training keeps the default foreach AdamW).

    Same parameters, same gradients, same moments: time foreach (current) vs fused=True and
    report the parameter difference after one step. Any adoption would be a separate decision."""
    device=torch.device('cuda');cfg=load_config('configs/v35/baseline.yaml');set_seed(42)
    model=LatentForestLM(cfg['model'],42).to(device);rows=[]
    g=torch.Generator(device=device).manual_seed(0)
    grads=[torch.randn(p.shape,device=device,generator=g)*1e-3 for p in model.parameters()]
    results={}
    for variant in ('foreach','fused'):
        set_seed(42);m=LatentForestLM(cfg['model'],42).to(device);m.load_state_dict(model.state_dict())
        groups={}
        for name,p in m.named_parameters():
            lr=cfg['train']['decoder_lr'] if name.startswith('decoder.') else cfg['train']['auxiliary_lr'];wd=cfg['train']['weight_decay'] if p.ndim>=2 else 0.
            groups.setdefault((lr,wd),[]).append(p)
        opt=torch.optim.AdamW([dict(params=ps,lr=lr,weight_decay=wd) for (lr,wd),ps in groups.items()],betas=tuple(cfg['train']['betas']),fused=variant=='fused')
        samples=[]
        for rep in range(8):
            for p,gr in zip(m.parameters(),grads):p.grad=gr.clone()
            torch.cuda.synchronize();a_,b_=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);tick=time.perf_counter();a_.record();opt.step();b_.record();torch.cuda.synchronize()
            samples.append(dict(wall_ms=(time.perf_counter()-tick)*1000,cuda_ms=a_.elapsed_time(b_)))
        results[variant]=dict(samples=samples,params={n:p.detach().clone() for n,p in m.named_parameters()},state=[(s['exp_avg'].clone(),s['exp_avg_sq'].clone()) for s in opt.state.values()])
        rows.append(dict(variant=variant,steps=8,wall_median_ms=statistics.median(x['wall_ms'] for x in samples),cuda_median_ms=statistics.median(x['cuda_ms'] for x in samples),first_step_ms=samples[0]['wall_ms']))
        del m,opt;gc.collect();torch.cuda.empty_cache()
    diff=max(float((results['foreach']['params'][n]-results['fused']['params'][n]).abs().max()) for n in results['foreach']['params'])
    rel=max(float(((results['foreach']['params'][n]-results['fused']['params'][n]).abs()/(results['foreach']['params'][n].abs()+1e-12)).max()) for n in results['foreach']['params'])
    out=dict(rows=rows,parameter_max_abs_difference_after_8_steps=diff,parameter_max_rel_difference=rel,identical=diff==0.,
             note='Diagnostic only. The fused kernel evaluates the same AdamW update; floating-point differences are reported, not hidden. Training/benchmarks in this report keep the default implementation.')
    atomic_json(out,root/'optimizer_check.json');print(json.dumps(out,default=str)[:800],flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--audit-dir',default='reports/cky_v35/audit');p.add_argument('--output-dir',default='reports/cky_v35/stage_d')
    p.add_argument('--cache-dir',default='cache/tinystories');p.add_argument('--micro-batches',nargs='+',type=int,default=[4,8]);p.add_argument('--steps',type=int,default=40)
    p.add_argument('--phases',nargs='+',choices=['build','stream','optimizer'],default=['build','stream','optimizer'])
    a=p.parse_args();root=Path(a.output_dir);root.mkdir(parents=True,exist_ok=True)
    if 'build' in a.phases:build_times(a,root)
    if 'stream' in a.phases:stream_replay(a,root)
    if 'optimizer' in a.phases:optimizer_check(a,root)
    print('stage D complete',flush=True)

if __name__=='__main__':main()
