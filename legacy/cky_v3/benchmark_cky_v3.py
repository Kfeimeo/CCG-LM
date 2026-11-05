"""Snapshot-paired Lazy/V3 CUDA benchmark. Diagnostic/profiler runs are separate."""
from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))  # repo root: run from the repo root or via python -m
del _sys, _Path
import argparse,cProfile,gc,json,math,pstats,statistics,subprocess,sys,time
from collections import Counter,defaultdict
from contextlib import contextmanager
from pathlib import Path
import pandas as pd
import torch
from legacy.cky_v2.benchmark_cky_v2 import CASES,cpu_snapshot,snapshot_hash
from legacy.cky_v25.benchmark_cky_v25 import autograd_nodes
from legacy.precision_v25.benchmark_precision_v25 import restore,step
from models.model import LatentForestLM
from models.precision_policy import POLICIES,configure_model,backend_state
from models.forest_reference import masked_log_softmax
from profile_performance import make_batch,staged_step
from train import make_optimizer
from utils import atomic_json,device_info,load_config,rng_state,set_seed,to_device


def timed(fn):
    a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize();tick=time.perf_counter();a.record();out=fn();b.record();b.synchronize()
    return out,dict(wall_ms=(time.perf_counter()-tick)*1000,cuda_envelope_ms=a.elapsed_time(b))

def stats(values):return dict(median=statistics.median(values),minimum=min(values),maximum=max(values))


def profile_operators(model,batch,opt,policy,scaler):
    contexts=[]
    def before(m,i):
        ctx=torch.profiler.record_function('CKY_FORWARD');ctx.__enter__();contexts.append(ctx)
    def after(m,i,o):contexts.pop().__exit__(None,None,None)
    handles=[model.forest.register_forward_pre_hook(before),model.forest.register_forward_hook(after)]
    original=model.forest._search_scores
    def search(*a,**kw):
        with torch.profiler.record_function('SCORE_SEARCH'):return original(*a,**kw)
    model.forest._search_scores=search;model.forest.v3_profile=True
    try:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
            _,timing=timed(lambda:step(model,batch,opt,policy,scaler))
    finally:
        for h in handles:h.remove()
        del model.forest._search_scores;model.forest.v3_profile=False
    counts=defaultdict(Counter);widths=Counter()
    for event in prof.events():
        if event.device_type!=torch.autograd.DeviceType.CPU:continue
        key=event.name
        if key.startswith('cudaLaunchKernel'):key='cudaLaunchKernel'
        if key not in ('cudaLaunchKernel','aten::stack','aten::cat','aten::select','aten::gather','aten::index_select'):continue
        counts['full'][key]+=1;parent=event.cpu_parent;anc=[]
        while parent is not None:anc.append(parent.name);parent=parent.cpu_parent
        if 'CKY_FORWARD' in anc:counts['cky'][key]+=1
        if 'SCORE_SEARCH' in anc:counts['score'][key]+=1
        if key=='cudaLaunchKernel':
            width=next((s for s in anc if s.startswith('V3/width_') and s.endswith('/elapsed_ms')),None)
            if width:widths[int(width.split('/')[1].split('_')[1])]+=1
    flattened={f'{scope}_{name.replace("aten::","")}':value[name] for scope,value in counts.items() for name in ('cudaLaunchKernel','aten::stack','aten::cat','aten::select','aten::gather','aten::index_select')}
    return flattened,dict(widths),timing


def run(case,args):
    name,config,size,words,real=case;set_seed(42);torch.set_num_threads(4)
    policy=POLICIES['B' if args.precision=='bf16' else 'E'];policy.configure_backend()
    cfg=load_config(f'configs/{config}.yaml');cfg['data']['cache_dir']=args.cache_dir
    model=configure_model(LatentForestLM(cfg['model']).cuda().train(),policy);opt=make_optimizer(model,cfg['train']);scaler=policy.make_scaler()
    batch=to_device(make_batch(cfg,size,4,words,128,'cache' if real else 'synthetic'),torch.device('cuda'))
    for p in model.parameters():p.grad=torch.zeros_like(p)
    opt.step();opt.zero_grad(set_to_none=True);torch.cuda.synchronize()
    state=dict(model=cpu_snapshot(model.state_dict()),optimizer=cpu_snapshot(opt.state_dict()),scaler=scaler.state_dict(),rng=rng_state())
    sha=snapshot_hash(state['model']);input_sha=snapshot_hash({k:v.cpu() for k,v in batch.items() if isinstance(v,torch.Tensor)})
    arms=(args.reference,args.implementation)
    def reset(impl):restore(model,opt,scaler,state,impl)
    for impl in arms:
        reset(impl)
        for _ in range(args.warmup):assert not step(model,batch,opt,policy,scaler)
    samples={x:[] for x in arms};peaks={x:[] for x in arms};orders=[]
    for repeat in range(args.repeats):
        order=arms if repeat%2==0 else tuple(reversed(arms));orders.append(order)
        for impl in order:
            reset(impl);torch.cuda.reset_peak_memory_stats()
            skipped,t=timed(lambda:step(model,batch,opt,policy,scaler));assert not skipped
            samples[impl].append(t);peaks[impl].append(torch.cuda.max_memory_allocated()/2**30)
    sample_root=Path(args.output_dir)/'performance'/args.precision/name;sample_root.mkdir(parents=True,exist_ok=True)
    atomic_json(dict(samples=samples,peaks=peaks,orders=orders,weight_sha256=sha,input_sha256=input_sha),sample_root/'uninstrumented_samples.json')
    rows=[];operator_rows=[];width_rows=[];details={};common_upstream=None;expected_work=None
    for impl in arms:
        parts=[]
        for _ in range(3):reset(impl);parts.append(staged_step(model,batch,opt,policy.outer_autocast))
        audits=[];shapes=[];captured={}
        reset(impl);model.forest.collect_timing=True
        def capture_input(m,i):captured['input']=tuple(x.detach() for x in i[:4])
        def capture_output(m,i,o):captured['prior']=o['log_prior']
        handles=[model.forest.register_forward_pre_hook(capture_input),model.forest.register_forward_hook(capture_output),
                 model.composer.register_forward_pre_hook(lambda m,i:shapes.append(list(i[0].shape)))]
        with policy.outer_context('cuda'):out=model(**batch)
        assert torch.isfinite(out['loss']) and torch.isfinite(out['logits']).all()
        audit=model.forest.last_workload.copy();audits.append(audit)
        graphs=dict(score_autograd_nodes=autograd_nodes([captured['prior']]),forest_autograd_nodes=autograd_nodes([out['memory'][k] for k in ('semantic','vector_coeff','log_prior')]))
        upstream=torch.autograd.grad(out['loss'],captured['prior'])[0].detach()
        if common_upstream is None:common_upstream=upstream
        else:assert torch.equal(common_upstream,upstream),'Forward/Reader gradients differ at fixed identical roots'
        for h in handles:h.remove()
        model.forest.collect_timing=False
        assert audit['v25_actual_composer_rows']==sum(math.prod(x[:-1]) for x in shapes)
        work=(audit['v25_actual_composer_rows'],audit['v25_composer_calls'])
        if expected_work is None:expected_work=work
        else:assert work==expected_work,'V3 changed lazy Composer work'
        leaves,potentials,coeff,mask=captured['input'];del out,captured
        search_samples=[]
        for _ in range(3):
            reset(impl);pot=potentials.clone().requires_grad_()
            (search,prior),tf=timed(lambda:(lambda s:(s,masked_log_softmax(s['root_s'],s['root_m'])))(model.forest._search_scores(leaves,pot,mask)))
            graphs['isolated_score_autograd_nodes']=autograd_nodes([prior])
            _,tb=timed(lambda:prior.backward(common_upstream))
            assert pot.grad is not None and torch.isfinite(pot.grad).all()
            search_samples.append(dict(forward=tf,backward=tb));del search,prior,pot
        reset(impl);counts,kernels,prof_timing=profile_operators(model,batch,opt,policy,scaler)
        operator_rows.append(dict(case=name,precision=args.precision,implementation=impl,**counts,**graphs))
        for item in audit.get('width_profile',[]):
            width_rows.append(dict(case=name,precision=args.precision,implementation=impl,**item,kernel_launches=kernels.get(item['width'],0)))
        reset(impl);python=cProfile.Profile();python.enable();step(model,batch,opt,policy,scaler);torch.cuda.synchronize();python.disable()
        ps=pstats.Stats(python);project=str(Path.cwd()).lower()
        python_self=sum(v[2] for key,v in ps.stats.items() if key[0].lower().startswith(project))*1000
        wall=[x['wall_ms'] for x in samples[impl]];tokens=int(batch['labels'].ne(-100).sum())
        row=dict(case=name,precision=args.precision,implementation=impl,batch=size,words_actual=batch['history_ids'].shape[-1],target_length=batch['labels'].shape[-1],ce_tokens=tokens,
            weight_sha256=sha,input_sha256=input_sha,step_median_ms=statistics.median(wall),step_min_ms=min(wall),step_max_ms=max(wall),ce_tokens_per_second=tokens*1000/statistics.median(wall),peak_allocated_gib=max(peaks[impl]),samples=len(wall),
            composer_rows=work[0],composer_calls=work[1],python_project_self_ms=python_self,nan_inf_audit=0,nonfinite_gradient_steps=0,**counts,**graphs)
        for label,key in [('forest_forward','module/cky'),('forest_backward','phase/cky_backward')]:
            values=[x[key].get('wall_ms',x[key]['cuda_envelope_ms']) for x in parts]
            for stat,value in stats(values).items():row[f'{label}_{stat}_ms']=value
        for label,kind in [('score_forward','forward'),('score_backward','backward')]:
            for stat,value in stats([x[kind]['wall_ms'] for x in search_samples]).items():row[f'{label}_{stat}_ms']=value
        row.update(score_forward_envelope_ms=audit['score_search_cuda_ms'],lazy_adapter_host_ms=audit['topology_d2h_blocking_ms']+audit['dag_cpu_ms']+audit['index_h2d_host_ms'],
                   lazy_semantic_envelope_ms=audit['semantic_cuda_ms'],index_cache_hit=audit.get('index_cache_hit'),index_cache_bytes=audit.get('index_cache_bytes'),
                   index_cache_total_build_host_ms=getattr(getattr(model.forest,'_geometry_cache',None),'build_host_ms',None) if impl!='lazy_semantic' else None)
        details[impl]=dict(samples=samples[impl],staged=parts,workload=audit,score_samples=search_samples,profiled_timing=prof_timing,python_self_scope='cProfile sum of project Python exclusive wall time; not OS thread CPU time',backend=backend_state())
        rows.append(row);print(json.dumps(row),flush=True)
    ratios=[a['wall_ms']/b['wall_ms'] for a,b in zip(samples[arms[0]],samples[arms[1]])]
    root=Path(args.output_dir)/'performance'/args.precision/name;root.mkdir(parents=True,exist_ok=True)
    atomic_json(dict(orders=orders,paired_speedup=stats(ratios),paired_ratio_samples=ratios,arms=details,policy=policy.state_dict()),root/'samples.json')
    pd.DataFrame(rows).to_csv(root/'benchmark.csv',index=False);pd.DataFrame(operator_rows).to_csv(root/'operator_counts.csv',index=False)
    pd.DataFrame(width_rows).to_csv(root/'width_profile.csv',index=False)
    del model,opt,state,batch;gc.collect();torch.cuda.empty_cache()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',default='reports/cky_v3')
    p.add_argument('--cache-dir',default='cache/tinystories_integration');p.add_argument('--implementation',choices=['lazy_tensorized_a','lazy_tensorized'],default='lazy_tensorized');p.add_argument('--reference',choices=['lazy_semantic','lazy_tensorized_a'],default='lazy_semantic')
    p.add_argument('--precision',choices=['bf16','fp32'],default='bf16');p.add_argument('--warmup',type=int,default=2);p.add_argument('--repeats',type=int,default=8)
    p.add_argument('--worker',action='store_true');p.add_argument('--case',choices=[x[0] for x in CASES]);p.add_argument('--cases',nargs='+',choices=[x[0] for x in CASES]);p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    if a.reference==a.implementation:p.error('Reference and implementation must differ')
    if a.warmup<2 or a.repeats<8:p.error('Require warmup>=2 and samples>=8')
    root=Path(a.output_dir);(root/'logs').mkdir(parents=True,exist_ok=True)
    env_path=root/f'environment_{a.precision}.json'
    if a.resume and env_path.exists():
        previous=json.loads(env_path.read_text(encoding='utf-8'))
        if previous['environment']!=device_info():raise ValueError('Resume environment differs; use a new output directory')
    if a.worker:return run(next(c for c in CASES if c[0]==a.case),a)
    for case in CASES:
        if a.cases and case[0] not in a.cases:continue
        existing=root/'performance'/a.precision/case[0]/'benchmark.csv'
        if a.resume and existing.exists():
            if set(pd.read_csv(existing).implementation)!={a.reference,a.implementation}:raise ValueError('Resume arm mismatch')
            continue
        cmd=[sys.executable,__file__,'--worker','--case',case[0],'--output-dir',str(root),'--cache-dir',a.cache_dir,'--implementation',a.implementation,'--reference',a.reference,'--precision',a.precision,'--warmup',str(a.warmup),'--repeats',str(a.repeats)]
        print(a.implementation,a.precision,case[0],flush=True)
        with (root/'logs'/f'benchmark_{a.precision}_{case[0]}.log').open('w',encoding='utf-8') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
        if r.returncode:raise RuntimeError('Case failed; raw log retained: '+case[0])
        for filename in ('benchmark.csv','operator_counts.csv','width_profile.csv'):
            paths=list((root/'performance').glob('*/*/'+filename))
            nonempty=[pd.read_csv(x) for x in paths if x.stat().st_size>2]
            pd.concat(nonempty,ignore_index=True).to_csv(root/filename,index=False)
    atomic_json(dict(environment=device_info(),arguments=vars(a),precision_policy=POLICIES['B' if a.precision=='bf16' else 'E'].state_dict(),protocol='Same physical model, common zero-grad AdamW setup, alternating arms, identical snapshot/RNG restore; input resident; timings exclude diagnostic/profiler/setup'),root/f'environment_{a.precision}.json')

if __name__=='__main__':main()
