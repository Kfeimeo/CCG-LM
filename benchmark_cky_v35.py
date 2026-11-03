"""V3.5 snapshot-paired multi-arm CUDA benchmark: V3 eager vs V3-C vs local torch.compile arms.

Protocol (per case, own subprocess): one physical model (seed 42), common zero-grad
AdamW initialization, identical model/optimizer/RNG snapshot restored before EVERY
sample, arms rotated each repeat, inputs resident, timings include forward, loss,
backward, clip and AdamW. Correctness (fixed weights, same RNG) and component
diagnostics (isolated score forward/backward, autograd nodes, profiler launches and
kernel-active time) are measured separately from the uninstrumented samples.
Compile arms report first-call compile time, warm steady state, recompiles and
CUDA-graph usage separately; nothing is excluded from the stream measurement.
"""
from __future__ import annotations
import argparse,gc,json,math,statistics,subprocess,sys,time
from collections import Counter
from pathlib import Path
import pandas as pd
import torch
from benchmark_cky_v2 import cpu_snapshot,snapshot_hash
from benchmark_cky_v25 import autograd_nodes
from models.model import LatentForestLM
from models.precision_policy import POLICIES,configure_model,backend_state
from models.forest_reference import masked_log_softmax
from models.forest_tensorized import SCORE_RECONSTRUCT
from models.cky_compile import compile_counters,reset_compile_state,configure_inductor
from profile_performance import make_batch
from reader_precision_study import tensor_errors,global_gradients
from v3_study import compare_search,assert_rng_equal
from train import make_optimizer
from utils import atomic_json,device_info,load_config,restore_rng,rng_state,set_seed,to_device

ARMS={
    'v3':dict(implementation='lazy_tensorized',backend='eager',mode='dag'),
    'v3c':dict(implementation=SCORE_RECONSTRUCT,backend='eager',mode='dag'),
    'v3c_tree':dict(implementation=SCORE_RECONSTRUCT,backend='eager',mode='tree'),
    'v3c_linear':dict(implementation=SCORE_RECONSTRUCT,backend='eager',mode='linear'),
    'v3_function':dict(implementation='lazy_tensorized',backend='function',mode='dag'),
    'v3_compile_default':dict(implementation='lazy_tensorized',backend='compile_default',mode='dag'),
    'v3_compile_ro':dict(implementation='lazy_tensorized',backend='compile_reduce_overhead',mode='dag'),
    'v3c_compile_default':dict(implementation=SCORE_RECONSTRUCT,backend='compile_default',mode='dag'),
    'v3c_compile_ro':dict(implementation=SCORE_RECONSTRUCT,backend='compile_reduce_overhead',mode='dag'),
}
# name, config, batch, words, kind ('synthetic' | 'real_integration' | 'real_main')
CASES=[
    ('smoke_b2_w8','smoke',2,8,'synthetic'),('smoke_b2_w16','smoke',2,16,'synthetic'),('smoke_b2_w24','smoke',2,24,'synthetic'),('smoke_b8_w24','smoke',8,24,'synthetic'),
    ('main_b1_w24','main_125m',1,24,'synthetic'),('main_b4_w24','main_125m',4,24,'synthetic'),('main_b8_w24','main_125m',8,24,'synthetic'),('main_b16_w24','main_125m',16,24,'synthetic'),
    ('real_b2','smoke',2,24,'real_integration'),
    ('real_main_mb4','main_125m',4,None,'real_main'),('real_main_mb16','main_125m',16,None,'real_main'),('real_main_mb32','main_125m',32,None,'real_main'),
]


def timed(fn):
    a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize();tick=time.perf_counter();a.record();out=fn();b.record();b.synchronize()
    return out,dict(wall_ms=(time.perf_counter()-tick)*1000,cuda_envelope_ms=a.elapsed_time(b))

def stats(values):return dict(median=statistics.median(values),minimum=min(values),maximum=max(values))


def set_arm(model,arm):
    spec=ARMS[arm];model.forest.implementation=spec['implementation'];model.forest.search_backend=spec['backend'];model.forest.reconstruction_mode=spec['mode']


def restore(model,opt,state,arm):
    opt.zero_grad(set_to_none=True);model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);restore_rng(state['rng']);set_arm(model,arm);torch.cuda.synchronize()


def step(model,batch,opt,policy,clip=1.):
    opt.zero_grad(set_to_none=True)
    with policy.outer_context('cuda'):out=model(**batch)
    out['loss'].backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),clip,error_if_nonfinite=True);opt.step()


def real_main_batch(cfg,mbs):
    from data.dataset import StoryDataset
    from profile_cky_v35 import probe_batches
    data=StoryDataset(cfg['data']['cache_dir']);return probe_batches(data,mbs,4096,1,skip_updates=2)[0]


def profile_step(model,batch,opt,policy):
    contexts=[]
    def before(m,i):
        ctx=torch.profiler.record_function('CKY_FORWARD');ctx.__enter__();contexts.append(ctx)
    def after(m,i,o):contexts.pop().__exit__(None,None,None)
    handles=[model.forest.register_forward_pre_hook(before),model.forest.register_forward_hook(after)]
    original=model.forest._search_scores
    def search(*a,**kw):
        with torch.profiler.record_function('SCORE_SEARCH'):return original(*a,**kw)
    model.forest._search_scores=search
    try:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
            _,timing=timed(lambda:step(model,batch,opt,policy))
    finally:
        for h in handles:h.remove()
        del model.forest._search_scores
    launches=Counter();graph_launches=0;device=[];sync_calls=Counter()
    for e in prof.events():
        if e.device_type!=torch.autograd.DeviceType.CPU:
            if not (getattr(e,'is_user_annotation',False) or e.name in ('CKY_FORWARD','SCORE_SEARCH')):device.append((e.time_range.start,e.time_range.end))
            continue
        if e.name.startswith('cudaGraphLaunch'):graph_launches+=1
        if e.name in ('cudaStreamSynchronize','cudaDeviceSynchronize','cudaMemcpyAsync','cudaMemcpy'):sync_calls[e.name]+=1
        if not e.name.startswith('cudaLaunchKernel'):continue
        launches['full']+=1;parent=e.cpu_parent;anc=set()
        while parent is not None:anc.add(parent.name);parent=parent.cpu_parent
        if 'CKY_FORWARD' in anc:launches['cky']+=1
        if 'SCORE_SEARCH' in anc:launches['score']+=1
    device.sort();busy=0.;cur=None
    for s,e in device:
        if cur is None or s>cur[1]:
            if cur:busy+=cur[1]-cur[0]
            cur=[s,e]
        else:cur[1]=max(cur[1],e)
    if cur:busy+=cur[1]-cur[0]
    return dict(full_cuda_launches=launches['full'],cky_forward_launches=launches['cky'],score_search_launches=launches['score'],cuda_graph_launches=graph_launches,
                profiled_wall_ms=timing['wall_ms'],profiled_cuda_envelope_ms=timing['cuda_envelope_ms'],gpu_busy_union_ms=busy/1000,gpu_idle_ms=timing['cuda_envelope_ms']-busy/1000,
                gpu_idle_fraction=1-(busy/1000)/timing['cuda_envelope_ms'],runtime_sync_calls=dict(sync_calls))


def correctness(model,batch,state,policy,opt,arm,reference_out,reference_grads,reference_rng_end,precision):
    restore(model,opt,state,arm);model.forest.capture_spans=True;start=rng_state()
    with policy.outer_context('cuda'):out=model(**batch)
    end=rng_state();out['loss'].backward()
    grads={n:None if p.grad is None else p.grad.detach().cpu().clone() for n,p in model.named_parameters()}
    snap=cpu_snapshot(out);model.forest.capture_spans=False;del out
    if reference_out is None:return snap,grads,end,None
    report=dict(arm=arm,rng_equal=True,search_identity=True,roots_bitwise=True,logits_bitwise=bool(torch.equal(snap['logits'],reference_out['logits'])),loss_bitwise=bool(torch.equal(snap['loss'],reference_out['loss'])))
    try:assert_rng_equal(reference_rng_end,end)
    except AssertionError:report['rng_equal']=False
    try:compare_search(reference_out['memory'],snap['memory'],precision=='bf16')
    except AssertionError as exc:
        report['search_identity']=False;report['roots_bitwise']=False;report['search_error']=str(exc)[:600]
        changed=0;total=0
        for span,x in reference_out['memory']['spans'].items():
            y=snap['memory']['spans'][span]
            for key in ('selected_flat','retained_flat','candidate_mask','scores'):
                if x[key] is not None:changed+=int(x[key].ne(y[key]).sum());total+=x[key].numel()
        report['span_entries_changed']=changed;report['span_entries_total']=total
    report['forward']={k:tensor_errors(reference_out[k],snap[k]) for k in ('logits','loss')}
    report['root']={k:tensor_errors(reference_out['memory'][k],snap['memory'][k]) for k in ('semantic','scores','vector_coeff','log_prior')}
    report['gradient']=global_gradients(reference_grads,grads,strict_label=precision)
    report['gradient'].pop('modules_detail',None)
    return snap,grads,end,report


def cap_allocator(margin_gib=0.25):
    free,total=torch.cuda.mem_get_info();cap=max(0.1,(free-margin_gib*2**30)/total);torch.cuda.set_per_process_memory_fraction(cap)
    print(json.dumps(dict(free_device_memory_gib=free/2**30,allocator_cap_gib=cap*total/2**30)),flush=True)


def run(case,args):
    name,config,size,words,kind=case;set_seed(42);torch.set_num_threads(4);cap_allocator()
    policy=POLICIES['B' if args.precision=='bf16' else 'E'];policy.configure_backend()
    cfg=load_config(f'configs/{config}.yaml')
    cfg['data']['cache_dir']=args.cache_dir if kind=='real_main' else args.integration_cache_dir
    cfg['model']['forest_implementation']='lazy_tensorized'
    model=configure_model(LatentForestLM(cfg['model']).cuda().train(),policy);opt=make_optimizer(model,cfg['train'])
    if kind=='real_main':batch=real_main_batch(cfg,size)
    else:batch=make_batch(cfg,size,4,words,128,'cache' if kind=='real_integration' else 'synthetic')
    batch=to_device(batch,torch.device('cuda'))
    for p in model.parameters():p.grad=torch.zeros_like(p)
    opt.step();opt.zero_grad(set_to_none=True);torch.cuda.synchronize()
    state=dict(model=cpu_snapshot(model.state_dict()),optimizer=cpu_snapshot(opt.state_dict()),rng=rng_state())
    sha=snapshot_hash(state['model']);input_sha=snapshot_hash({k:v.cpu() for k,v in batch.items() if isinstance(v,torch.Tensor)})
    arms=list(args.arms);tokens=int(batch['labels'].ne(-100).sum())
    shape=dict(batch=int(batch['input_ids'].shape[0]),history_slots=int(batch['history_ids'].shape[1]),words_actual=int(batch['history_ids'].shape[-1]),target_length=int(batch['labels'].shape[-1]),ce_tokens=tokens,
               valid_history_sentences=int(batch['history_mask'].sum()),valid_words=int(batch['word_mask'].sum()))
    compile_info={};reset_compile_state()
    # Warmup each arm; for compile arms the first call is the compilation cost (reported, not excluded from stream results).
    # Dynamo state is reset once per case so earlier compile arms keep their graphs during the rotated timing loop.
    for arm in arms:
        restore(model,opt,state,arm)
        if 'compile' in arm:
            settings=configure_inductor();gc.collect();torch.cuda.empty_cache();before=compile_counters()
            _,first=timed(lambda:step(model,batch,opt,policy));after_first=compile_counters()
            restore(model,opt,state,arm);_,second=timed(lambda:step(model,batch,opt,policy))
            for _ in range(max(0,args.warmup-2)):restore(model,opt,state,arm);step(model,batch,opt,policy)
            compile_info[arm]=dict(first_call_ms=first['wall_ms'],second_call_ms=second['wall_ms'],counters_before=before,counters_after_first=after_first,counters_after_warmup=compile_counters(),inductor_settings=settings)
        else:
            for _ in range(args.warmup):restore(model,opt,state,arm);step(model,batch,opt,policy)
    samples={x:[] for x in arms};peaks={x:[] for x in arms};orders=[]
    for repeat in range(args.repeats):
        order=arms[repeat%len(arms):]+arms[:repeat%len(arms)]
        if repeat%2:order=list(reversed(order))
        orders.append(order)
        for arm in order:
            restore(model,opt,state,arm);torch.cuda.reset_peak_memory_stats()
            _,t=timed(lambda:step(model,batch,opt,policy));samples[arm].append(t);peaks[arm].append(torch.cuda.max_memory_allocated()/2**30)
    for arm in arms:
        if arm in compile_info:compile_info[arm]['counters_after_timing']=compile_counters()
    root=Path(args.output_dir)/'performance'/args.precision/name;root.mkdir(parents=True,exist_ok=True)
    atomic_json(dict(samples=samples,peaks=peaks,orders=orders,weight_sha256=sha,input_sha256=input_sha,shape=shape,compile=compile_info),root/'uninstrumented_samples.json')
    # Correctness against the V3 eager reference with identical weights and RNG.
    reference=arms[0] if arms[0]=='v3' else 'v3'
    ref_out,ref_grads,ref_rng,_=correctness(model,batch,state,policy,opt,reference,None,None,None,args.precision)
    reports={};rows=[];details={}
    for arm in arms:
        if arm==reference:continue
        _,_,_,rep=correctness(model,batch,state,policy,opt,arm,ref_out,ref_grads,ref_rng,args.precision);reports[arm]=rep
    for arm in arms:
        restore(model,opt,state,arm);captured={}
        h1=model.forest.register_forward_pre_hook(lambda m,i:captured.__setitem__('input',tuple(x.detach() for x in i[:4])))
        h2=model.forest.register_forward_hook(lambda m,i,o:captured.__setitem__('prior',o['log_prior']))
        model.forest.collect_timing=True
        with policy.outer_context('cuda'):out=model(**batch)
        model.forest.collect_timing=False;h1.remove();h2.remove()
        workload=dict(model.forest.last_workload);workload.pop('width_profile',None)
        graphs=dict(score_autograd_nodes=autograd_nodes([captured['prior']]),forest_autograd_nodes=autograd_nodes([out['memory'][k] for k in ('semantic','vector_coeff','log_prior')]),loss_autograd_nodes=autograd_nodes([out['loss']]))
        upstream=torch.autograd.grad(out['loss'],captured['prior'],retain_graph=False)[0].detach()
        leaves,potentials,coeff,mask=captured['input'];del out,captured
        # Isolated: (a) search forward only; (b) prior->potential backward through the differentiable score path
        # (V3: search graph; V3-C: reconstruction graph). Both arms use the same upstream prior gradient.
        search_f=[];prior_b=[];forest_f=[]
        from contextlib import nullcontext
        search_ctx=torch.no_grad if ARMS[arm]['implementation']==SCORE_RECONSTRUCT else nullcontext
        for _ in range(3):
            restore(model,opt,state,arm);pot=potentials.clone().requires_grad_()
            def isolated_search():
                with search_ctx():return model.forest._search_scores(leaves,pot,mask)
            _,tf=timed(isolated_search);search_f.append(tf['wall_ms'])
            restore(model,opt,state,arm);pot=potentials.clone().requires_grad_()
            (res,),tff=timed(lambda:(model.forest(leaves,pot,coeff,mask),));forest_f.append(tff['wall_ms'])
            prior=res['log_prior']
            _,tb=timed(lambda:torch.autograd.grad(prior,pot,upstream));prior_b.append(tb['wall_ms']);del res,prior,pot
        restore(model,opt,state,arm);prof=profile_step(model,batch,opt,policy)
        wall=[x['wall_ms'] for x in samples[arm]]
        row=dict(case=name,precision=args.precision,arm=arm,**ARMS[arm],**shape,weight_sha256=sha,input_sha256=input_sha,
            step_median_ms=statistics.median(wall),step_min_ms=min(wall),step_max_ms=max(wall),ce_tokens_per_second=tokens*1000/statistics.median(wall),peak_allocated_gib=max(peaks[arm]),samples=len(wall),
            score_forward_median_ms=statistics.median(search_f),forest_forward_median_ms=statistics.median(forest_f),score_backward_median_ms=statistics.median(prior_b),
            reconstruction_cuda_ms=workload.get('score_reconstruction_cuda_ms'),reconstruction_host_ms=workload.get('score_reconstruction_host_ms'),reconstruction_nodes=workload.get('score_reconstruction_nodes'),
            d2h_blocking_ms=workload.get('topology_d2h_blocking_ms'),dag_cpu_ms=workload.get('dag_cpu_ms'),index_h2d_host_ms=workload.get('index_h2d_host_ms'),semantic_cuda_ms=workload.get('semantic_cuda_ms'),score_search_cuda_ms=workload.get('score_search_cuda_ms'),
            composer_rows=workload.get('v25_actual_composer_rows'),composer_calls=workload.get('v25_composer_calls'),index_cache_hit=workload.get('index_cache_hit'),**graphs,**prof)
        if arm in compile_info:
            ci=compile_info[arm];row.update(compile_first_call_ms=ci['first_call_ms'],compile_second_call_ms=ci['second_call_ms'],
                compiled_frames=(ci['counters_after_warmup'].get('frames',{}).get('ok',0)-ci['counters_before'].get('frames',{}).get('ok',0)),graph_breaks=sum(ci['counters_after_warmup'].get('graph_break',{}).values())-sum(ci['counters_before'].get('graph_break',{}).values()),
                recompiles_after_warmup=sum(ci['counters_after_warmup'].get('recompiles',{}).values()) if 'recompiles' in ci['counters_after_warmup'] else None,
                recompiles_during_timing=(sum(ci['counters_after_timing'].get('frames',{}).values())-sum(ci['counters_after_warmup'].get('frames',{}).values())),
                cudagraph_skips=ci['counters_after_warmup'].get('inductor',{}).get('cudagraph_skips',0))
        if arm in reports and reports[arm] is not None:
            rep=reports[arm];row.update(search_identity=rep['search_identity'],rng_equal=rep['rng_equal'],logits_bitwise=rep['logits_bitwise'],loss_bitwise=rep['loss_bitwise'],
                root_scores_max_abs=rep['root']['scores']['max_absolute_error'],log_prior_max_abs=rep['root']['log_prior']['max_absolute_error'],semantic_max_abs=rep['root']['semantic']['max_absolute_error'],
                loss_max_abs=rep['forward']['loss']['max_absolute_error'],grad_strict_failures=rep['gradient']['strict_failed_parameter_count'],grad_max_abs=rep['gradient']['global_statistics']['max_absolute_gradient_error'],
                grad_rel_l2=rep['gradient']['global_statistics'].get('relative_l2_error'),span_scorer_grad_norm=rep['gradient']['modules']['span_scorer']['gradient_norm_lazy'])
        details[arm]=dict(samples=samples[arm],workload=workload,search_forward_ms=search_f,prior_backward_ms=prior_b,forest_forward_ms=forest_f,profile=prof,correctness=reports.get(arm),compile=compile_info.get(arm))
        rows.append(row);print(json.dumps({k:v for k,v in row.items() if k in ('case','arm','step_median_ms','ce_tokens_per_second','score_backward_median_ms','full_cuda_launches','score_autograd_nodes','search_identity','grad_strict_failures','compile_first_call_ms')}),flush=True)
    ratios={arm:stats([a['wall_ms']/b['wall_ms'] for a,b in zip(samples[reference],samples[arm])]) for arm in arms if arm!=reference}
    atomic_json(dict(orders=orders,paired_speedup_vs_reference=ratios,arms=details,policy=policy.state_dict(),backend=backend_state(),shape=shape),root/'samples.json')
    pd.DataFrame(rows).to_csv(root/'benchmark.csv',index=False)
    del model,opt,state,batch;gc.collect();torch.cuda.empty_cache()


def stream(args):
    """Consecutive REAL micro-batches of varying shape: total time including every (re)compile."""
    set_seed(42);torch.set_num_threads(4);cap_allocator();policy=POLICIES['B'];policy.configure_backend()
    cfg=load_config('configs/main_125m.yaml');cfg['data']['cache_dir']=args.cache_dir;cfg['model']['forest_implementation']='lazy_tensorized'
    from data.dataset import StoryDataset
    from profile_cky_v35 import probe_batches
    data=StoryDataset(cfg['data']['cache_dir']);batches=probe_batches(data,args.stream_micro_batch,4096,args.stream_steps,skip_updates=3)
    model=configure_model(LatentForestLM(cfg['model']).cuda().train(),policy);opt=make_optimizer(model,cfg['train'])
    for p in model.parameters():p.grad=torch.zeros_like(p)
    opt.step();opt.zero_grad(set_to_none=True);state=dict(model=cpu_snapshot(model.state_dict()),optimizer=cpu_snapshot(opt.state_dict()),rng=rng_state())
    rows=[];details={}
    for arm in args.arms:
        restore(model,opt,state,arm)
        if 'compile' in arm:reset_compile_state();configure_inductor()
        if hasattr(model.forest,'_geometry_cache'):model.forest._geometry_cache.clear()
        gc.collect();torch.cuda.empty_cache();per=[];shapes=[];before=compile_counters()
        torch.cuda.synchronize();tick=time.perf_counter()
        for b in batches:
            gpu=to_device(b,torch.device('cuda'));_,t=timed(lambda:step(model,gpu,opt,policy));per.append(t['wall_ms']);shapes.append((int(gpu['history_ids'].shape[1])*int(gpu['history_ids'].shape[0]),int(gpu['history_ids'].shape[-1])));del gpu
        torch.cuda.synchronize();total=(time.perf_counter()-tick)*1000;after=compile_counters()
        frames=int(after.get('stats',{}).get('unique_graphs',0))-int(before.get('stats',{}).get('unique_graphs',0))  # Dynamo graphs captured (one per width x shape); inductor fxgraph_cache_miss counts fwd+bwd FX graphs
        fx_compiled=int(after.get('inductor',{}).get('fxgraph_cache_miss',0))-int(before.get('inductor',{}).get('fxgraph_cache_miss',0))
        cache=model.forest._geometry_cache.statistics() if hasattr(model.forest,'_geometry_cache') else None
        row=dict(arm=arm,**ARMS[arm],micro_batch_size=args.stream_micro_batch,steps=len(batches),total_wall_ms=total,median_step_ms=statistics.median(per),first_step_ms=per[0],
                 last_quarter_median_ms=statistics.median(per[-max(1,len(per)//4):]),unique_shapes=len(set(shapes)),compiled_frames=frames if 'compile' in arm else 0,fx_graphs_compiled=fx_compiled if 'compile' in arm else 0,compile_seconds_est=sum(x for x in per if x>2000)/1000 if 'compile' in arm else 0.,
                 cache_misses=cache['misses'] if cache else None,cache_hit_rate=cache['hit_rate'] if cache else None,cache_build_ms=cache['build_host_ms_total'] if cache else None,
                 ce_tokens=sum(int(b['labels'].ne(-100).sum()) for b in batches))
        row['ce_tokens_per_second']=row['ce_tokens']*1000/total;rows.append(row);details[arm]=dict(per_step_ms=per,shapes=shapes,counters=after)
        print(json.dumps(row),flush=True)
    root=Path(args.output_dir)/'stream';root.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(root/f'stream_mb{args.stream_micro_batch}.csv',index=False);atomic_json(details,root/f'stream_mb{args.stream_micro_batch}.json')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',default='reports/cky_v35/benchmark')
    p.add_argument('--cache-dir',default='cache/tinystories');p.add_argument('--integration-cache-dir',default='cache/tinystories_integration')
    p.add_argument('--arms',nargs='+',choices=list(ARMS),default=['v3','v3c']);p.add_argument('--precision',choices=['bf16','fp32'],default='bf16')
    p.add_argument('--warmup',type=int,default=3);p.add_argument('--repeats',type=int,default=10)
    p.add_argument('--worker',action='store_true');p.add_argument('--case',choices=[x[0] for x in CASES]);p.add_argument('--cases',nargs='+',choices=[x[0] for x in CASES])
    p.add_argument('--stream',action='store_true');p.add_argument('--stream-steps',type=int,default=30);p.add_argument('--stream-micro-batch',type=int,default=16)
    a=p.parse_args()
    if a.warmup<3 or a.repeats<10:p.error('Require warmup>=3 and repeats>=10')
    if a.arms[0]!='v3':p.error('First arm must be the v3 eager reference')
    root=Path(a.output_dir);(root/'logs').mkdir(parents=True,exist_ok=True)
    if a.worker:return run(next(c for c in CASES if c[0]==a.case),a)
    if a.stream:return stream(a)
    for case in CASES:
        if a.cases and case[0] not in a.cases:continue
        cmd=[sys.executable,__file__,'--worker','--case',case[0],'--output-dir',str(root),'--cache-dir',a.cache_dir,'--integration-cache-dir',a.integration_cache_dir,'--precision',a.precision,'--warmup',str(a.warmup),'--repeats',str(a.repeats),'--arms',*a.arms]
        print(a.precision,case[0],a.arms,flush=True)
        with (root/'logs'/f'benchmark_{a.precision}_{case[0]}.log').open('w',encoding='utf-8') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
        if r.returncode:print('CASE FAILED (log retained):',case[0],flush=True);continue
        paths=list((root/'performance').glob('*/*/benchmark.csv'))
        pd.concat([pd.read_csv(x) for x in paths if x.stat().st_size>2],ignore_index=True).to_csv(root/'benchmark.csv',index=False)
    atomic_json(dict(environment=device_info(),arguments=vars(a),protocol=__doc__),root/f'environment_{a.precision}.json')

if __name__=='__main__':main()
