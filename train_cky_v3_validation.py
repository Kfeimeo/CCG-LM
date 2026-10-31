"""8-step CPU trajectories and 20-step real-data CUDA checkpoint migration/replay."""
from __future__ import annotations
import argparse,copy,gc,hashlib,json,time
from pathlib import Path
import pandas as pd
import torch
from tqdm.auto import tqdm
from benchmark_cky_v2 import cpu_snapshot
from data.dataset import StoryDataset,StatefulSampler,Collator
from models.model import LatentForestLM
from models.precision_policy import POLICIES,configure_model
from train import make_optimizer,make_schedule,train_update,save_checkpoint,load_checkpoint
from train_precision_stability import FixedSubset
from validate_cky_v25 import small_config,model_batch
from precision_study import search_equal
from utils import atomic_json,rng_state,restore_rng,set_seed,to_device
from v3_study import assert_rng_equal,assert_state_equal


def diffs(a,b):
    return {n:float((p.detach()-dict(b.named_parameters())[n].detach()).abs().max()) for n,p in a.named_parameters()}


def cpu_case(mode,root):
    set_seed(2026,True);torch.set_num_threads(1);pp=POLICIES['E'];pp.configure_backend();cfg=small_config(mode)
    models=[configure_model(LatentForestLM(cfg['model']),pp) for _ in range(2)];models[1].load_state_dict(models[0].state_dict())
    for m,impl in zip(models,('lazy_semantic','lazy_tensorized')):m.forest.implementation=impl;m.forest.capture_spans=True
    opts=[make_optimizer(m,cfg['train']) for m in models];schedules=[make_schedule(o,cfg['train']) for o in opts];rows=[]
    for step in range(8):
        batch=model_batch(cfg);initial=rng_state();outputs=[];metrics=[];ends=[]
        for m,o,sch in zip(models,opts,schedules):
            restore_rng(initial);o.zero_grad(set_to_none=True);out=m(**batch);out['loss'].backward()
            metrics.append(dict(loss=float(out['loss'].detach()),scorer_norm=float(sum(p.grad.square().sum() for p in m.syntax.span_scorer.parameters() if p.grad is not None).sqrt()),
                                composer_norm=float(sum(p.grad.square().sum() for p in m.composer.parameters() if p.grad is not None).sqrt()),scalar_bias_grad=float(m.syntax.span_scorer.scalar.bias.grad)))
            assert metrics[-1]['scorer_norm']>0 and metrics[-1]['composer_norm']>0
            norm=float(torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True));o.step();sch.step(int(batch['labels'].ne(-100).sum()))
            metrics[-1]['grad_norm']=norm;metrics[-1]['learning_rates']=[g['lr'] for g in o.param_groups]
            outputs.append(cpu_snapshot(out['memory']));ends.append(rng_state());del out
        assert_rng_equal(*ends);delta=diffs(*models);structure=search_equal(*outputs)
        rows.append(dict(step=step+1,target_tokens=schedules[0].tokens,reference=metrics[0],v3=metrics[1],loss_error=abs(metrics[0]['loss']-metrics[1]['loss']),
                         search=structure,max_parameter_error=max(delta.values()),neutral_bias_error=delta['syntax.span_scorer.scalar.bias']))
    atomic_json(dict(mode=mode,device='cpu',precision='fp32',steps=8,trajectory=rows,parameter_errors=delta),root/f'cpu_{mode}.json')
    print(mode,'CPU 8 steps completed',flush=True)


def real_cuda(root,checkpoint,cache_dir,micro_batch_size=None):
    # Deterministic algorithms are used for the checkpoint replay proof, in both arms.
    set_seed(42,True);torch.set_num_threads(4);pp=POLICIES['B'];pp.configure_backend()
    old=torch.load(checkpoint,map_location='cpu',weights_only=False);cfg=copy.deepcopy(old['config'])
    source_micro_batch_size=cfg['train']['micro_batch_size']
    if micro_batch_size is not None:cfg['train']['micro_batch_size']=micro_batch_size
    data=FixedSubset(StoryDataset(cache_dir));assert data.fingerprint==old['data_fingerprint']
    begin=old['state']['step'];start_tokens=old['state']['target_tokens'];configs=[];models=[];opts=[];schedules=[];samplers=[];scalers=[]
    for impl in ('lazy_semantic','lazy_tensorized'):
        c=copy.deepcopy(cfg);c['model']['forest_implementation']=impl;c['precision']=pp.state_dict();c['train']['deterministic']=True;c['train']['budget_tokens']=start_tokens+20*c['train']['effective_tokens']
        m=configure_model(LatentForestLM(c['model'],c['seed']).cuda(),pp);o=make_optimizer(m,c['train']);sch=make_schedule(o,c['train']);sam=StatefulSampler(data,c['seed']);scaler=pp.make_scaler()
        load_checkpoint(checkpoint,m,o,sch,sam,c,data.fingerprint,restore_random=False,scaler=scaler)
        configs.append(c);models.append(m);opts.append(o);schedules.append(sch);samplers.append(sam);scalers.append(scaler)
    restore_rng(old['rng']);del old
    checkpoint_out=Path('runs/cky_v3')/(root.name+'_checkpoint_510.pt');rows=[];replay={};durations=[0.,0.];tokens=cfg['train']['effective_tokens']
    probe=to_device(Collator()([data[i] for i in range(2)]),torch.device('cuda'))
    progress=tqdm(range(20),desc='V3 real checkpoint +20',ascii=True,mininterval=2)
    for step in progress:
        initial=rng_state();metrics=[None,None];ends=[None,None]
        for idx in ([0,1] if step%2==0 else [1,0]):
            restore_rng(initial);torch.cuda.synchronize();tick=time.perf_counter()
            metrics[idx]=train_update(models[idx],samplers[idx],opts[idx],schedules[idx],tokens,cfg['train']['micro_batch_size'],torch.device('cuda'),pp.outer_autocast,cfg['train']['grad_clip'],scaler=scalers[idx])
            torch.cuda.synchronize();seconds=time.perf_counter()-tick;metrics[idx]['seconds']=seconds;durations[idx]+=seconds;ends[idx]=rng_state()
            assert metrics[idx]['span_scorer_grad_norm']>0 and metrics[idx]['composer_grad_norm']>0
        assert_rng_equal(*ends);assert samplers[0].state_dict()==samplers[1].state_dict();assert schedules[0].state_dict()==schedules[1].state_dict()
        delta=diffs(*models);row=dict(step=begin+step+1,target_tokens=start_tokens+(step+1)*tokens,reference=metrics[0],v3=metrics[1],
            loss_error=abs(metrics[0]['train_nll']-metrics[1]['train_nll']),max_parameter_error=max(delta.values()),neutral_bias_error=delta['syntax.span_scorer.scalar.bias'],
            learning_rates=[g['lr'] for g in opts[1].param_groups],sampler=samplers[1].state_dict())
        if (step+1)%5==0:
            state=rng_state();mem=[]
            for m in models:
                restore_rng(state);m.forest.capture_spans=True
                with torch.no_grad(),pp.outer_context('cuda'):out=m(**probe)
                mem.append(cpu_snapshot(out['memory']));del out;m.forest.capture_spans=False
            row['probe_search']=search_equal(*mem);restore_rng(state)
        rows.append(row);progress.set_postfix(loss=f"{metrics[1]['train_nll']:.3f}")
        if step==9:
            save_checkpoint(checkpoint_out,models[1],opts[1],schedules[1],samplers[1],configs[1],dict(step=begin+10,target_tokens=start_tokens+10*tokens),data.fingerprint,scaler=scalers[1])
        if step==10:
            # Reconstruct from V3 checkpoint and replay the very same next update.
            state_after=rng_state();expected=dict(model=cpu_snapshot(models[1].state_dict()),optimizer=cpu_snapshot(opts[1].state_dict()),rng=state_after,sampler=samplers[1].state_dict(),schedule=schedules[1].state_dict())
            saved=load_checkpoint(checkpoint_out,models[1],opts[1],schedules[1],samplers[1],configs[1],data.fingerprint,scaler=scalers[1])
            assert saved['forest_implementation']=='lazy_tensorized'
            assert_state_equal(saved['optimizer'],cpu_snapshot(opts[1].state_dict()))
            for key,value in saved['model'].items():assert torch.equal(value.cpu(),models[1].state_dict()[key].cpu())
            replay_metrics=train_update(models[1],samplers[1],opts[1],schedules[1],tokens,cfg['train']['micro_batch_size'],torch.device('cuda'),pp.outer_autocast,cfg['train']['grad_clip'],scaler=scalers[1])
            actual=cpu_snapshot(models[1].state_dict())
            maximum=max(float((actual[n]-v).abs().max()) for n,v in expected['model'].items())
            assert_state_equal(expected['optimizer'],cpu_snapshot(opts[1].state_dict()))
            assert maximum==0,maximum;assert_rng_equal(expected['rng'],rng_state())
            assert expected['sampler']==samplers[1].state_dict() and expected['schedule']==schedules[1].state_dict()
            assert replay_metrics['train_nll']==metrics[1]['train_nll']
            replay=dict(from_step=begin+10,replayed_step=begin+11,parameter_max_error=maximum,rng_equal=True,scheduler_equal=True,sampler_equal=True,optimizer_loaded=True,optimizer_replayed_bitwise=True)
            restore_rng(state_after)
    atomic_json(dict(source_checkpoint=str(checkpoint),source_step=begin,end_step=begin+20,source_tokens=start_tokens,end_tokens=start_tokens+20*tokens,
        steps=20,model='smoke',effective_ce_tokens=tokens,micro_batch_size=cfg['train']['micro_batch_size'],source_micro_batch_size=source_micro_batch_size,source_data_fingerprint=data.fingerprint,source_examples=data.indices,precision=pp.state_dict(),deterministic_replay=True,
        elapsed_training_seconds=durations,ce_tokens_per_second=[20*tokens/x for x in durations],
        timing_scope='Actual mmap sampling/collation, H2D, exact CE accumulation, forward/backward/clip/AdamW; excludes validation/checkpoint/probes; one short sequential pair, not an official pretraining throughput claim',
        trajectory=rows,checkpoint_replay=replay,parameter_errors=delta),root/'cuda_real_training.json')
    print('Real CUDA +20 and checkpoint replay completed',flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',default='reports/cky_v3/training');p.add_argument('--cpu-only',action='store_true');p.add_argument('--cuda-only',action='store_true')
    p.add_argument('--checkpoint',default='reports/cky_v25_precision/stability_runs/long/B2_seed42/checkpoint.pt');p.add_argument('--cache-dir',default='cache/tinystories_integration');p.add_argument('--micro-batch-size',type=int)
    a=p.parse_args();root=Path(a.output_dir);root.mkdir(parents=True,exist_ok=True)
    if not a.cuda_only:
        for mode in ('forest_scalar4','forest_vector4'):cpu_case(mode,root)
    if not a.cpu_only:real_cuda(root,Path(a.checkpoint),a.cache_dir,a.micro_batch_size)

if __name__=='__main__':main()
