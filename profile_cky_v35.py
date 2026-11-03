"""V3.5 Stage A: real-data training performance audit of the V3 lazy_tensorized forest.

Everything here is measurement. No model mathematics, search budget, loss, data or
optimizer semantics are modified. The audited update mirrors train.train_update
(exact SUM-loss accumulation over an effective CE-token budget, clip, AdamW,
scheduler step) with host timers and CUDA events added around each stage.

Three timing views are reported and must not be added together as a precise
decomposition: host wall time (CPU issue/blocking), CUDA event envelopes
(GPU-timeline span, includes launch gaps and waits) and profiler kernel-active
time (sum/union of device kernel durations, measured in a separate profiled run).
"""
from __future__ import annotations
import argparse,cProfile,gc,json,math,pstats,statistics,sys,time
from collections import Counter,OrderedDict,defaultdict
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from data.dataset import Collator,StatefulSampler,StoryDataset
from models.cky_index_cache import CKYIndexCache
from models.model import LatentForestLM,parameter_report
from models.precision_policy import PrecisionPolicy,backend_state,configure_model
from models.forest_reference import masked_log_softmax
from benchmark_cky_v25 import autograd_nodes
from profile_performance import staged_step
from train import make_optimizer,make_schedule
from utils import atomic_json,device_info,load_config,restore_rng,rng_state,set_seed,sha256,to_device

PROBE_SEED=7350001  # Performance-probe sampler seed: not a training seed (13/42/2026), not the dev split.
MICRO_BATCHES=[1,2,4,8,16,32,64,128]


def cuda_pair():
    return torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)


class StageTimer:
    """Host wall + CUDA envelope per named stage; envelopes resolved after a final sync."""
    def __init__(self):self.records=[]
    @contextmanager
    def stage(self,name,sync=False):
        if sync:torch.cuda.synchronize()
        a,b=cuda_pair();tick=time.perf_counter();a.record()
        try:yield
        finally:
            b.record();host=(time.perf_counter()-tick)*1000
            if sync:torch.cuda.synchronize();host=(time.perf_counter()-tick)*1000
            self.records.append((name,host,a,b))
    def hook(self,module,name):
        stack=[]
        def before(m,args):
            a,b=cuda_pair();tick=time.perf_counter();a.record();stack.append((a,b,tick))
        def after(m,args,out):
            a,b,tick=stack.pop();b.record();self.records.append((name,(time.perf_counter()-tick)*1000,a,b))
        return [module.register_forward_pre_hook(before),module.register_forward_hook(after)]
    def totals(self):
        torch.cuda.synchronize();out=defaultdict(lambda:dict(host_ms=0.,cuda_envelope_ms=0.,calls=0))
        for name,host,a,b in self.records:
            d=out[name];d['host_ms']+=host;d['cuda_envelope_ms']+=a.elapsed_time(b);d['calls']+=1
        return dict(out)


def build_model(cfg,device):
    policy=PrecisionPolicy(**cfg['precision']);policy.configure_backend();set_seed(cfg['seed'])
    model=configure_model(LatentForestLM(cfg['model'],cfg['seed']).to(device).train(),policy)
    return model,policy


def weight_fingerprint(model):
    import hashlib
    digest=hashlib.sha256()
    for name,tensor in model.state_dict().items():digest.update(name.encode());digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def batch_shape_stats(batch):
    wm=batch['word_mask'];hm=batch['history_mask'];labels=batch['labels']
    b,h,w=wm.shape;lengths=wm.sum(-1)
    valid_sentences=int(hm.sum());valid_words=int(wm.sum())
    hist_lengths=lengths[hm].tolist()
    return dict(batch=b,history_slots=h,padded_words=w,target_length=int(labels.shape[1]),ce_tokens=int(labels.ne(-100).sum()),
        valid_history_sentences=valid_sentences,history_sentence_lengths=hist_lengths,
        word_padding_ratio=1-valid_words/max(1,b*h*w),history_slot_padding_ratio=1-valid_sentences/max(1,b*h),
        target_padding_ratio=1-int(labels.ne(-100).sum())/max(1,labels.numel()),
        span_utilization=sum(l*(l+1)/2 for l in hist_lengths)/max(1,b*h*w*(w+1)/2),
        cache_key=(w,b*h))


def dedup_stats(batch):
    ids=batch['history_ids'];wm=batch['word_mask'];hm=batch['history_mask'];seen=Counter()
    for r in range(ids.shape[0]):
        for s in range(ids.shape[1]):
            if hm[r,s]:
                n=int(wm[r,s].sum());seen[tuple(ids[r,s,:n].tolist())]+=1
    occurrences=sum(seen.values());return dict(history_occurrences=occurrences,unique_histories=len(seen),duplicate_fraction=1-len(seen)/max(1,occurrences))


def audited_update(model,sampler,optimizer,scheduler,tokens,mbs,device,policy,clip,hooks=True,forest_timing=False,sync_stages=False):
    """train.train_update semantics with timers. SUM loss over exact token budget, clip, AdamW."""
    model.train();optimizer.zero_grad(set_to_none=True);timer=StageTimer();handles=[]
    if hooks:
        for module,name in [(model.syntax,'module/syntax'),(model.syntax.span_scorer,'module/span_scorer'),(model.forest,'module/forest'),(model.decoder,'module/decoder')]:
            handles+=timer.hook(module,name)
        for reader in model.readers.values():handles+=timer.hook(reader,'module/reader')
    model.forest.collect_timing=forest_timing
    done=0;loss_sum=0.;micro=[];forest_workloads=[]
    try:
        while done<tokens:
            t0=time.perf_counter();batch,count=sampler.batch(mbs,tokens-done);t1=time.perf_counter()
            shape=batch_shape_stats(batch)
            with timer.stage('h2d',sync=sync_stages):gpu=to_device(batch,device)
            with timer.stage('forward',sync=sync_stages):
                with policy.outer_context(device):result=model(**gpu);loss=result['loss_sum']/tokens
            with timer.stage('backward',sync=sync_stages):loss.backward()
            loss_sum+=float(result['loss_sum'].detach());done+=count
            shape.update(sample_ms=(t1-t0)*1000)
            if forest_timing:forest_workloads.append({k:v for k,v in model.forest.last_workload.items() if k!='width_profile'}|dict(width_profile=model.forest.last_workload.get('width_profile')))
            micro.append(shape);del result,loss,gpu
        with timer.stage('clip',sync=sync_stages):norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),clip,error_if_nonfinite=True))
        with timer.stage('optimizer',sync=sync_stages):optimizer.step()
        scheduler.step(done)
    finally:
        for h in handles:h.remove()
        model.forest.collect_timing=False
    totals=timer.totals()
    return dict(train_nll=loss_sum/done,step_tokens=done,micro_batches=len(micro),grad_norm=norm,stages=totals,micro=micro,forest_workloads=forest_workloads)


def fresh_state(model,optimizer):
    from benchmark_cky_v2 import cpu_snapshot
    return dict(model=cpu_snapshot(model.state_dict()),optimizer=cpu_snapshot(optimizer.state_dict()),rng=rng_state())


def restore_state(model,optimizer,state):
    optimizer.zero_grad(set_to_none=True);model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);restore_rng(state['rng']);torch.cuda.synchronize()


def sweep(args,cfg,root,data):
    """Real-data micro-batch sweep at a fixed effective CE-token budget."""
    device=torch.device('cuda');model,policy=build_model(cfg,device);optimizer=make_optimizer(model,cfg['train'])
    for p in model.parameters():p.grad=torch.zeros_like(p)
    optimizer.step();optimizer.zero_grad(set_to_none=True);state=fresh_state(model,optimizer);sha=weight_fingerprint(model)
    tokens=args.effective_tokens;clip=cfg['train']['grad_clip'];rows=[];details={}
    for mbs in args.micro_batches:
        restore_state(model,optimizer,state);scheduler=make_schedule(optimizer,cfg['train']);sampler=StatefulSampler(data,PROBE_SEED)
        if hasattr(model.forest,'_geometry_cache'):model.forest._geometry_cache.clear()
        warm=args.warmup if mbs>=8 else 1;reps=args.repeats if mbs>=8 else max(3,args.repeats//2)
        gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();updates=[];status='ok';error=None
        try:
            for i in range(warm+reps):
                torch.cuda.synchronize();tick=time.perf_counter()
                metrics=audited_update(model,sampler,optimizer,scheduler,tokens,mbs,device,policy,clip,hooks=i>=warm,forest_timing=False)
                torch.cuda.synchronize();wall=(time.perf_counter()-tick)*1000
                if i>=warm:
                    updates.append(dict(wall_ms=wall,micro_batches=metrics['micro_batches'],step_tokens=metrics['step_tokens'],train_nll=metrics['train_nll'],grad_norm=metrics['grad_norm'],
                        stages={k:dict(host_ms=v['host_ms'],cuda_envelope_ms=v['cuda_envelope_ms'],calls=v['calls']) for k,v in metrics['stages'].items()},
                        sample_ms=sum(m['sample_ms'] for m in metrics['micro']),micro_shapes=[{k:v for k,v in m.items() if k not in ('history_sentence_lengths',)} for m in metrics['micro']],
                        cache=model.forest._geometry_cache.statistics() if hasattr(model.forest,'_geometry_cache') else None))
                print(json.dumps(dict(mbs=mbs,update=i,wall_ms=round(wall,1),micro_batches=metrics['micro_batches'],tokens=metrics['step_tokens'])),flush=True)
        except torch.OutOfMemoryError as exc:
            status='oom';error=str(exc).splitlines()[0][:300];print('OOM at micro-batch',mbs,flush=True)
            optimizer.zero_grad(set_to_none=True)
        peak=torch.cuda.max_memory_allocated()/2**30
        if updates:
            walls=[u['wall_ms'] for u in updates];toks=[u['step_tokens'] for u in updates];mbc=[u['micro_batches'] for u in updates]
            stage_names=sorted({k for u in updates for k in u['stages']})
            row=dict(micro_batch_size=mbs,status=status,updates=len(updates),effective_tokens=tokens,update_wall_median_ms=statistics.median(walls),update_wall_min_ms=min(walls),update_wall_max_ms=max(walls),
                micro_batches_per_update_median=statistics.median(mbc),micro_batch_wall_median_ms=statistics.median(u['wall_ms']/u['micro_batches'] for u in updates),
                ce_tokens_per_second=sum(toks)/(sum(walls)/1000),sampling_collation_ms_per_update=statistics.median(u['sample_ms'] for u in updates),peak_allocated_gib=peak,
                mean_padded_words=statistics.mean(s['padded_words'] for u in updates for s in u['micro_shapes']),mean_history_slots=statistics.mean(s['history_slots'] for u in updates for s in u['micro_shapes']),
                mean_target_length=statistics.mean(s['target_length'] for u in updates for s in u['micro_shapes']),mean_word_padding_ratio=statistics.mean(s['word_padding_ratio'] for u in updates for s in u['micro_shapes']),
                mean_target_padding_ratio=statistics.mean(s['target_padding_ratio'] for u in updates for s in u['micro_shapes']),cache_hit_rate_cumulative=updates[-1]['cache']['hit_rate'] if updates[-1]['cache'] else None,
                cache_misses_cumulative=updates[-1]['cache']['misses'] if updates[-1]['cache'] else None,cache_build_ms_cumulative=updates[-1]['cache']['build_host_ms_total'] if updates[-1]['cache'] else None)
            for name in stage_names:
                row[f'{name}_host_ms_per_update']=statistics.median(u['stages'][name]['host_ms'] for u in updates if name in u['stages'])
                row[f'{name}_cuda_ms_per_update']=statistics.median(u['stages'][name]['cuda_envelope_ms'] for u in updates if name in u['stages'])
        else:row=dict(micro_batch_size=mbs,status=status,updates=0,effective_tokens=tokens,peak_allocated_gib=peak,error=error)
        rows.append(row);details[str(mbs)]=dict(status=status,error=error,updates=updates,peak_allocated_gib=peak)
        pd.DataFrame(rows).to_csv(root/'sweep.csv',index=False);atomic_json(dict(weight_sha256=sha,probe_seed=PROBE_SEED,effective_tokens=tokens,details=details),root/'sweep_updates.json')
        gc.collect();torch.cuda.empty_cache()
    del model,optimizer,state;gc.collect();torch.cuda.empty_cache()


def probe_batches(data,mbs,tokens,count,skip_updates=0):
    """Deterministic real micro-batches from the probe stream (same examples every call)."""
    sampler=StatefulSampler(data,PROBE_SEED);batches=[]
    for _ in range(skip_updates):
        done=0
        while done<tokens:_,c=sampler.batch(mbs,tokens-done);done+=c
    while len(batches)<count:
        done=0
        while done<tokens and len(batches)<count:
            b,c=sampler.batch(mbs,tokens-done);done+=c;batches.append(b)
    return batches


def record_function_hooks(module,name):
    stack=[]
    def before(m,args):
        ctx=torch.profiler.record_function(name);ctx.__enter__();stack.append(ctx)
    def after(m,args,out):stack.pop().__exit__(None,None,None)
    return [module.register_forward_pre_hook(before),module.register_forward_hook(after)]


def profiler_attribution(prof,ranges_prefix=('module/','phase/','forest/','V3/','step')):
    """Attribute CPU op self-device time and launches to the innermost enclosing timestamp range."""
    events=list(prof.events());ranges=[]
    for e in events:
        if e.device_type==torch.autograd.DeviceType.CPU and e.name.startswith(ranges_prefix):
            ranges.append((e.time_range.start,e.time_range.end,e.name))
    ranges.sort(key=lambda r:(r[0],-(r[1]-r[0])))
    def innermost(t):
        best=None
        for s,en,name in ranges:
            if s<=t<=en and (best is None or en-s<best[0]):best=(en-s,name)
        return best[1] if best else 'unattributed'
    per=defaultdict(lambda:dict(kernel_active_ms=0.,launches=0,cpu_ops=0,cpu_self_ms=0.))
    device_intervals=[]
    def annotation(e):return getattr(e,'is_user_annotation',False) or e.name.startswith(ranges_prefix) or e.name.startswith('ProfilerStep')
    for e in events:
        if e.device_type!=torch.autograd.DeviceType.CPU:
            if not annotation(e):device_intervals.append((e.time_range.start,e.time_range.end))
            continue
        name=innermost(e.time_range.start);d=per[name]
        if e.name.startswith('cudaLaunchKernel'):d['launches']+=1
        if e.name.startswith('aten::') or e.name.startswith('cudaLaunchKernel'):
            d['cpu_ops']+=1;d['cpu_self_ms']+=e.self_cpu_time_total/1000
        d['kernel_active_ms']+=e.self_device_time_total/1000
    device_intervals.sort();busy=0.;cur=None
    for s,e in device_intervals:
        if cur is None or s>cur[1]:
            if cur:busy+=cur[1]-cur[0]
            cur=[s,e]
        else:cur[1]=max(cur[1],e)
    if cur:busy+=cur[1]-cur[0]
    return dict(per),busy/1000,len(device_intervals)


def decompose(args,cfg,root,data):
    """Per-micro-batch time decomposition on real batches, three views, for selected micro-batch sizes."""
    device=torch.device('cuda');model,policy=build_model(cfg,device);optimizer=make_optimizer(model,cfg['train'])
    for p in model.parameters():p.grad=torch.zeros_like(p)
    optimizer.step();optimizer.zero_grad(set_to_none=True);state=fresh_state(model,optimizer);sha=weight_fingerprint(model);tokens=args.effective_tokens;clip=cfg['train']['grad_clip']
    results={};rows=[]
    original_search=model.forest._search_scores
    for mbs in args.decompose_micro_batches:
        batches=probe_batches(data,mbs,tokens,args.decompose_batches,skip_updates=2)
        per_batch=[]
        for bi,cpu_batch in enumerate(batches):
            try:
                batch=to_device(cpu_batch,device);shape=batch_shape_stats(cpu_batch);shape.pop('history_sentence_lengths');shape['dedup']=dedup_stats(cpu_batch)
                restore_state(model,optimizer,state)
                def plain_step():
                    optimizer.zero_grad(set_to_none=True)
                    with policy.outer_context(device):out=model(**batch);loss=out['loss_sum']/tokens
                    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),clip,error_if_nonfinite=True);optimizer.step()
                for _ in range(2):plain_step()
                # 1) Uninstrumented micro-batch steps (hooks off): wall and CUDA envelope.
                plain=[]
                for _ in range(args.decompose_repeats):
                    restore_state(model,optimizer,state);a,b=cuda_pair();torch.cuda.synchronize();tick=time.perf_counter();a.record();plain_step();b.record();torch.cuda.synchronize()
                    plain.append(dict(wall_ms=(time.perf_counter()-tick)*1000,cuda_envelope_ms=a.elapsed_time(b)))
                # 2) Hook/event instrumented, no internal sync: module envelopes + stage host times.
                hooked=[]
                for _ in range(3):
                    restore_state(model,optimizer,state);timer=StageTimer();handles=[]
                    for module,name in [(model.syntax,'module/syntax'),(model.syntax.span_scorer,'module/span_scorer'),(model.forest,'module/forest'),(model.decoder,'module/decoder'),(model.composer,'module/composer')]:handles+=timer.hook(module,name)
                    for reader in model.readers.values():handles+=timer.hook(reader,'module/reader')
                    def search_wrap(*a_,**k_):
                        with timer.stage('forest/search'):return original_search(*a_,**k_)
                    model.forest._search_scores=search_wrap
                    optimizer.zero_grad(set_to_none=True);torch.cuda.synchronize();tick=time.perf_counter()
                    with timer.stage('h2d'):gpu=to_device(cpu_batch,device)
                    with timer.stage('forward'):
                        with policy.outer_context(device):out=model(**gpu);loss=out['loss_sum']/tokens
                    with timer.stage('loss_scalar_read'):float(out['loss_sum'].detach())
                    with timer.stage('backward'):loss.backward()
                    with timer.stage('clip'):torch.nn.utils.clip_grad_norm_(model.parameters(),clip,error_if_nonfinite=True)
                    with timer.stage('optimizer'):optimizer.step()
                    torch.cuda.synchronize();wall=(time.perf_counter()-tick)*1000
                    for h in handles:h.remove()
                    del model.forest._search_scores
                    hooked.append(dict(wall_ms=wall,stages=timer.totals()))
                # 3) Forest internal timing (collect_timing syncs at the end of the forest forward) + D2H wait/copy split.
                internal=[]
                for sync_search in (False,True):
                    restore_state(model,optimizer,state);model.forest.collect_timing=True;waits=[]
                    if sync_search:
                        def search_sync(*a_,**k_):
                            out_=original_search(*a_,**k_);tick_=time.perf_counter();torch.cuda.synchronize();waits.append((time.perf_counter()-tick_)*1000);return out_
                        model.forest._search_scores=search_sync
                    with policy.outer_context(device):out=model(**batch)
                    torch.cuda.synchronize();wl=dict(model.forest.last_workload);wp=wl.pop('width_profile',None)
                    if sync_search:del model.forest._search_scores
                    model.forest.collect_timing=False
                    internal.append(dict(sync_before_d2h=sync_search,gpu_wait_before_d2h_ms=sum(waits),workload=wl,width_profile=wp))
                    del out
                # 4) Autograd graph sizes.
                restore_state(model,optimizer,state);captured={}
                h1=model.forest.register_forward_hook(lambda m,i,o:captured.__setitem__('prior',o['log_prior']))
                h2=model.syntax.register_forward_hook(lambda m,i,o:captured.__setitem__('scores',o[0]))
                with policy.outer_context(device):out=model(**batch)
                h1.remove();h2.remove()
                graphs=dict(score_prior_autograd_nodes=autograd_nodes([captured['prior']]),syntax_autograd_nodes=autograd_nodes([captured['scores']]),
                    forest_output_autograd_nodes=autograd_nodes([out['memory'][k] for k in ('semantic','vector_coeff','log_prior')]),loss_autograd_nodes=autograd_nodes([out['loss']]))
                del out,captured
                # 5) Staged (detached-boundary) step for backward split; synchronized phases.
                restore_state(model,optimizer,state);staged=[staged_step(model,batch,optimizer,policy.outer_autocast,measure_composer_backward=True,measure_reader_backward=True) for _ in range(2)]
                # 6) Profiler: kernel-active attribution by innermost range, launch counts, GPU busy union.
                restore_state(model,optimizer,state);model.forest.v3_profile=True
                def search_prof(*a_,**k_):
                    with torch.profiler.record_function('forest/search'):return original_search(*a_,**k_)
                model.forest._search_scores=search_prof;handles=[]
                for module,name in [(model.syntax,'module/syntax'),(model.forest,'module/forest'),(model.decoder,'module/decoder'),(model.composer,'module/composer')]+[(rd,'module/reader') for rd in model.readers.values()]:
                    handles+=record_function_hooks(module,name)
                optimizer.zero_grad(set_to_none=True);torch.cuda.synchronize()
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
                    with torch.profiler.record_function('step'):
                        a,b=cuda_pair();tick=time.perf_counter();a.record()
                        with torch.profiler.record_function('phase/h2d'):gpu=to_device(cpu_batch,device)
                        with torch.profiler.record_function('phase/forward'):
                            with policy.outer_context(device):out=model(**gpu);loss=out['loss_sum']/tokens
                        with torch.profiler.record_function('phase/backward'):loss.backward()
                        with torch.profiler.record_function('phase/clip'):torch.nn.utils.clip_grad_norm_(model.parameters(),clip,error_if_nonfinite=True)
                        with torch.profiler.record_function('phase/optimizer'):optimizer.step()
                        b.record();torch.cuda.synchronize();prof_wall=(time.perf_counter()-tick)*1000;prof_env=a.elapsed_time(b)
                for h in handles:h.remove()
                del model.forest._search_scores;model.forest.v3_profile=False
                attribution,gpu_busy_ms,device_events=profiler_attribution(prof)
                launches=sum(1 for e in prof.events() if e.device_type==torch.autograd.DeviceType.CPU and e.name.startswith('cudaLaunchKernel'))
                memcpy=[e for e in prof.events() if e.device_type!=torch.autograd.DeviceType.CPU and 'memcpy' in e.name.lower()]
                memcpy_d2h_cpu=[e for e in prof.events() if e.device_type==torch.autograd.DeviceType.CPU and e.name in ('cudaMemcpyAsync','cudaStreamSynchronize','cudaDeviceSynchronize','cudaMemcpy')]
                top_kernels=Counter()
                for e in prof.events():
                    if e.device_type!=torch.autograd.DeviceType.CPU and not (getattr(e,'is_user_annotation',False) or e.name.startswith(('module/','phase/','forest/','V3/','step'))):top_kernels[e.name]+=e.time_range.end-e.time_range.start
                top=[dict(kernel=k[:120],device_ms=v/1000) for k,v in top_kernels.most_common(25)]
                # Profiled staged step for backward phase kernel attribution.
                restore_state(model,optimizer,state)
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof2:
                    staged_step(model,batch,optimizer,policy.outer_autocast,measure_composer_backward=True,measure_reader_backward=True)
                staged_attr,staged_busy,_=profiler_attribution(prof2)
                # 7) Python exclusive time: project vs torch vs other.
                restore_state(model,optimizer,state);py=cProfile.Profile();torch.cuda.synchronize();py.enable();plain_step();torch.cuda.synchronize();py.disable()
                ps=pstats.Stats(py);project=str(Path.cwd()).lower();buckets=Counter()
                for (file,line,func),v in ps.stats.items():
                    f=file.lower()
                    key='project' if f.startswith(project) else 'torch' if 'site-packages\\torch' in f or 'site-packages/torch' in f else 'builtin' if file.startswith('~') or file.startswith('<') else 'other'
                    buckets[key]+=v[2]*1000
                del prof,prof2,gpu,out,loss
                record=dict(micro_batch_size=mbs,batch_index=bi,shape=shape,plain=plain,hooked=hooked,forest_internal=internal,graphs=graphs,staged=staged,
                    profiler=dict(wall_ms=prof_wall,cuda_envelope_ms=prof_env,gpu_busy_union_ms=gpu_busy_ms,device_events=device_events,cuda_launches=launches,memcpy_count=len(memcpy),
                                  memcpy_ms=sum(e.time_range.end-e.time_range.start for e in memcpy)/1000,runtime_sync_calls={n:sum(1 for e in memcpy_d2h_cpu if e.name==n) for n in ('cudaMemcpyAsync','cudaStreamSynchronize','cudaDeviceSynchronize','cudaMemcpy')},
                                  runtime_sync_cpu_ms={n:sum(e.self_cpu_time_total for e in memcpy_d2h_cpu if e.name==n)/1000 for n in ('cudaMemcpyAsync','cudaStreamSynchronize','cudaDeviceSynchronize','cudaMemcpy')},attribution=attribution,top_kernels=top,staged_attribution=staged_attr,staged_gpu_busy_ms=staged_busy),
                    python_exclusive_ms=dict(buckets))
                per_batch.append(record)
                plain_wall=statistics.median(x['wall_ms'] for x in plain);plain_env=statistics.median(x['cuda_envelope_ms'] for x in plain)
                row=dict(micro_batch_size=mbs,batch_index=bi,ce_tokens=shape['ce_tokens'],padded_words=shape['padded_words'],history_slots=shape['history_slots'],target_length=shape['target_length'],
                    plain_wall_median_ms=plain_wall,plain_cuda_envelope_median_ms=plain_env,ce_tokens_per_second_microbatch=shape['ce_tokens']*1000/plain_wall,
                    profiled_wall_ms=prof_wall,profiled_gpu_busy_ms=gpu_busy_ms,profiled_gpu_idle_fraction=1-gpu_busy_ms/prof_env,cuda_launches=launches,**{f'graph_{k}':v for k,v in graphs.items()},
                    python_project_ms=buckets['project'],python_torch_ms=buckets['torch'],python_builtin_ms=buckets['builtin'])
                hk=hooked[0]['stages']
                for name,v in sorted(hk.items()):row[f'{name}_host_ms']=statistics.median(h['stages'][name]['host_ms'] for h in hooked);row[f'{name}_cuda_ms']=statistics.median(h['stages'][name]['cuda_envelope_ms'] for h in hooked)
                wl=internal[0]['workload']
                for key in ('score_search_host_ms','score_search_cuda_ms','topology_d2h_blocking_ms','dag_cpu_ms','index_h2d_host_ms','index_h2d_cuda_ms','semantic_host_ms','semantic_cuda_ms','v25_actual_composer_rows','v25_composer_calls','index_cache_hit'):
                    row[f'forest_{key}']=wl.get(key)
                row['d2h_gpu_wait_ms_when_synced']=internal[1]['gpu_wait_before_d2h_ms'];row['d2h_copy_tolist_ms_when_synced']=internal[1]['workload']['topology_d2h_blocking_ms']
                for name,v in attribution.items():row[f'kernel_active_{name}_ms']=v['kernel_active_ms'];row[f'launches_{name}']=v['launches']
                for name,v in staged_attr.items():
                    if name.startswith('phase/') or name.endswith('_backward'):row[f'staged_kernel_active_{name}_ms']=v['kernel_active_ms']
                for name in staged[0]:
                    row[f'staged_{name}_cuda_ms']=statistics.median(s[name]['cuda_envelope_ms'] for s in staged)
                    if 'wall_ms' in staged[0][name]:row[f'staged_{name}_wall_ms']=statistics.median(s[name]['wall_ms'] for s in staged)
                rows.append(row);print(json.dumps(dict(mbs=mbs,batch=bi,plain_wall_ms=round(plain_wall,1),gpu_busy_ms=round(gpu_busy_ms,1),launches=launches)),flush=True)
            except torch.OutOfMemoryError as exc:
                print('OOM in decomposition at micro-batch',mbs,flush=True);per_batch.append(dict(micro_batch_size=mbs,batch_index=bi,status='oom',error=str(exc).splitlines()[0][:300]));optimizer.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache();break
        results[str(mbs)]=per_batch
        pd.DataFrame(rows).to_csv(root/'decomposition.csv',index=False);atomic_json(dict(weight_sha256=sha,probe_seed=PROBE_SEED,effective_tokens=tokens,results=results,
            notes='plain=uninstrumented micro-batch step (fwd+bwd+clip+AdamW, inputs resident); hooked=CUDA-event module envelopes without internal sync; forest_internal uses collect_timing (syncs at forest end); staged detaches graph boundaries and synchronizes phases; profiler numbers come from separate profiled runs and carry profiler overhead; kernel_active is CPU-op self device time attributed to the innermost timestamp range; gpu_busy_union is the union of device event intervals.'),root/'decomposition.json')
    del model,optimizer,state;gc.collect();torch.cuda.empty_cache()


def length_and_cache_analysis(args,cfg,root,data):
    """CPU-only data statistics over the probe stream + LRU key simulation; GPU build-time per unique key."""
    tokens=args.effective_tokens;report={};cache_rows=[];length_rows=[]
    model_cfg=cfg['model'];k,r=model_cfg['candidates'],model_cfg['max_compositions']
    unique_keys=set()
    for mbs in args.micro_batches:
        sampler=StatefulSampler(data,PROBE_SEED);keys=[];shapes=[];dedup=Counter();hist_lengths=Counter();target_lengths=Counter();valid_hist=Counter();updates=0;micro=0
        span_util=[];word_pad=[];slot_pad=[];target_pad=[];ce=[]
        while updates<args.stream_updates:
            done=0
            while done<tokens:
                b,c=sampler.batch(mbs,tokens-done);done+=c;micro+=1;s=batch_shape_stats(b);keys.append(s['cache_key']);shapes.append((s['batch'],s['history_slots'],s['padded_words'],s['target_length']))
                for l in s['history_sentence_lengths']:hist_lengths[l]+=1
                valid_hist[s['valid_history_sentences']]+=1
                for t in b['labels'].ne(-100).sum(-1).tolist():target_lengths[t]+=1
                d=dedup_stats(b);dedup['occurrences']+=d['history_occurrences'];dedup['unique']+=d['unique_histories']
                span_util.append(s['span_utilization']);word_pad.append(s['word_padding_ratio']);slot_pad.append(s['history_slot_padding_ratio']);target_pad.append(s['target_padding_ratio']);ce.append(s['ce_tokens'])
            updates+=1
        unique_keys.update(keys)
        sim={}
        for cap in (8,16,32,64,128):
            lru=OrderedDict();hits=misses=evictions=0;miss_keys=Counter()
            for key in keys:
                if key in lru:hits+=1;lru.move_to_end(key)
                else:
                    misses+=1;miss_keys[key]+=1;lru[key]=True
                    if len(lru)>cap:lru.popitem(last=False);evictions+=1
            sim[cap]=dict(hits=hits,misses=misses,evictions=evictions,hit_rate=hits/len(keys),cold_unique_keys=len(set(keys)),most_missed=[dict(words=kk[0],rows=kk[1],misses=v) for kk,v in miss_keys.most_common(8)])
        freq=Counter(keys)
        report[str(mbs)]=dict(updates=updates,micro_batches=micro,micro_batches_per_update=micro/updates,unique_keys=len(freq),key_frequency=[dict(words=kk[0],rows=kk[1],count=v,fraction=v/len(keys)) for kk,v in freq.most_common(20)],
            lru=sim,history_sentence_length_histogram=dict(sorted(hist_lengths.items())),valid_history_count_histogram=dict(sorted(valid_hist.items())),target_length_histogram=dict(sorted(target_lengths.items())),
            padded_words_mean=statistics.mean(s[2] for s in shapes),padded_words_histogram=dict(sorted(Counter(s[2] for s in shapes).items())),history_slots_mean=statistics.mean(s[1] for s in shapes),
            target_length_padded_mean=statistics.mean(s[3] for s in shapes),word_padding_ratio_mean=statistics.mean(word_pad),history_slot_padding_ratio_mean=statistics.mean(slot_pad),target_padding_ratio_mean=statistics.mean(target_pad),
            span_utilization_mean=statistics.mean(span_util),ce_tokens_per_micro_batch_mean=statistics.mean(ce),
            dedup=dict(history_occurrences=dedup['occurrences'],unique_histories=dedup['unique'],duplicate_fraction=1-dedup['unique']/max(1,dedup['occurrences'])))
        for cap,v in sim.items():cache_rows.append(dict(micro_batch_size=mbs,capacity=cap,**{kk:vv for kk,vv in v.items() if kk!='most_missed'}))
        length_rows.append(dict(micro_batch_size=mbs,micro_batches_per_update=micro/updates,padded_words_mean=report[str(mbs)]['padded_words_mean'],history_slots_mean=report[str(mbs)]['history_slots_mean'],
            target_length_padded_mean=report[str(mbs)]['target_length_padded_mean'],word_padding_ratio_mean=statistics.mean(word_pad),span_utilization_mean=statistics.mean(span_util),target_padding_ratio_mean=statistics.mean(target_pad),
            duplicate_history_fraction=report[str(mbs)]['dedup']['duplicate_fraction'],unique_cache_keys=len(freq),lru8_hit_rate=sim[8]['hit_rate'],lru32_hit_rate=sim[32]['hit_rate']))
        print('length/cache stream',mbs,'unique keys',len(freq),'lru8 hit',round(sim[8]['hit_rate'],3),flush=True)
    # Device build time per unique key actually observed (sorted by size); same geometry function as training.
    builds=[];device=torch.device('cuda')
    for (w,n) in sorted(unique_keys,key=lambda x:(x[1],x[0])):
        if n>512:continue
        samples=[]
        for _ in range(2):
            torch.cuda.synchronize();tick=time.perf_counter();plan=CKYIndexCache._build(w,n,k,r,device,full_decode=True);torch.cuda.synchronize();samples.append((time.perf_counter()-tick)*1000)
        builds.append(dict(words=w,rows=n,build_wall_ms_min=min(samples),build_wall_ms_max=max(samples),device_bytes=plan.device_bytes,size=plan.size));del plan
    torch.cuda.empty_cache()
    report['builds']=builds
    atomic_json(dict(probe_seed=PROBE_SEED,effective_tokens=tokens,stream_updates=args.stream_updates,k=k,r=r,per_micro_batch=report,
        notes='Cache key=(padded words, B*padded history slots) exactly as CKYIndexCache (layout/k/r/device/dtype fixed). LRU simulated on the real probe key stream. Builds timed on device with sync (host planning + H2D).'),root/'cache_analysis.json')
    pd.DataFrame(cache_rows).to_csv(root/'cache_simulation.csv',index=False);pd.DataFrame(length_rows).to_csv(root/'length_distribution.csv',index=False)
    atomic_json({m:{kk:vv for kk,vv in v.items() if kk in ('history_sentence_length_histogram','valid_history_count_histogram','target_length_histogram','padded_words_histogram','dedup','span_utilization_mean','word_padding_ratio_mean')} for m,v in report.items() if m!='builds'},root/'length_distribution.json')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='configs/v3_main_125m.yaml');p.add_argument('--cache-dir',default='cache/tinystories')
    p.add_argument('--output-dir',default='reports/cky_v35/audit');p.add_argument('--phases',nargs='+',choices=['sweep','decompose','stream'],default=['stream','sweep','decompose'])
    p.add_argument('--micro-batches',nargs='+',type=int,default=MICRO_BATCHES);p.add_argument('--decompose-micro-batches',nargs='+',type=int,default=[4,16,32])
    p.add_argument('--effective-tokens',type=int,default=4096);p.add_argument('--warmup',type=int,default=2);p.add_argument('--repeats',type=int,default=5)
    p.add_argument('--decompose-batches',type=int,default=3);p.add_argument('--decompose-repeats',type=int,default=5);p.add_argument('--stream-updates',type=int,default=100)
    p.add_argument('--memory-margin-gib',type=float,default=0.25,help='Cap the caching allocator at (free device memory - margin) so non-fitting configurations raise OOM instead of spilling into Windows shared system memory')
    a=p.parse_args();root=Path(a.output_dir);root.mkdir(parents=True,exist_ok=True)
    if not torch.cuda.is_available():p.error('CUDA required')
    free,total=torch.cuda.mem_get_info();cap=max(0.1,(free-a.memory_margin_gib*2**30)/total);torch.cuda.set_per_process_memory_fraction(cap)
    memory_state=dict(free_device_memory_gib_at_start=free/2**30,total_device_memory_gib=total/2**30,allocator_cap_fraction=cap,allocator_cap_gib=cap*total/2**30,alloc_conf=__import__('os').environ.get('PYTORCH_CUDA_ALLOC_CONF'))
    print(json.dumps(memory_state),flush=True)
    torch.set_num_threads(4);cfg=load_config(a.config);cfg['data']['cache_dir']=a.cache_dir
    data=StoryDataset(a.cache_dir)
    env=dict(environment=device_info(),gpu_properties=dict(name=torch.cuda.get_device_properties(0).name,total_memory_gib=torch.cuda.get_device_properties(0).total_memory/2**30,multiprocessors=torch.cuda.get_device_properties(0).multi_processor_count,capability=list(torch.cuda.get_device_capability(0))),
        python=sys.version,arguments=vars(a),config=cfg,precision_policy=PrecisionPolicy(**cfg['precision']).state_dict(),data_fingerprint=data.manifest['fingerprint'],dataset_stats=data.manifest['stats'],
        memory_state=memory_state,weights='seed-42 initialization of configs/v3_main_125m.yaml (no intact trained main checkpoint exists locally: runs/main_vector_seed42/*.pt are truncated archives)',
        source_hashes={str(x):sha256(x) for x in [Path('train.py'),*Path('models').glob('*.py'),Path(__file__)]})
    atomic_json(env,root/f'environment_{"_".join(a.phases)}.json')
    if 'stream' in a.phases:length_and_cache_analysis(a,cfg,root,data)
    if 'sweep' in a.phases:sweep(a,cfg,root,data)
    if 'decompose' in a.phases:decompose(a,cfg,root,data)
    env['backend']=backend_state();atomic_json(env,root/f'environment_{"_".join(a.phases)}.json');print('audit complete',flush=True)

if __name__=='__main__':main()
