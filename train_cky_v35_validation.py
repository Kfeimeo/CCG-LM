"""Short real-data training comparison (100-300 optimizer steps) between forest arms.

Each arm trains a fresh main model from the SAME seed-42 initialization on the SAME
real TinyStories example order (StatefulSampler seed), the same effective CE-token
budget per update, the same fixed validation subset and the same precision policy.
Arms run in separate processes (memory/compile isolation). Uses train.train_update
unchanged; only synchronized wall-clock timers are added around it.
"""
from __future__ import annotations
import argparse,hashlib,json,math,subprocess,sys,time
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader
from data.dataset import Collator,StatefulSampler,StoryDataset
from models.model import LatentForestLM
from models.precision_policy import PrecisionPolicy,configure_model,backend_state
from models.cky_compile import configure_inductor,compile_counters
from train import make_optimizer,make_schedule,train_update,save_checkpoint
from benchmark_cky_v35 import ARMS,set_arm
from utils import atomic_json,device_info,load_config,set_seed,to_device

TRAIN_SEED=42


def weight_hash(model):
    d=hashlib.sha256()
    for k,v in model.state_dict().items():d.update(k.encode());d.update(v.detach().cpu().contiguous().numpy().tobytes())
    return d.hexdigest()


@torch.no_grad()
def validate(model,batches,policy,device):
    model.eval();loss=tokens=0.;roots=histories=entropy=0.;nonfinite=0
    for batch in batches:
        with policy.outer_context(device):out=model(**batch)
        loss+=float(out['loss_sum']);tokens+=int(batch['labels'].ne(-100).sum())
        mem=out['memory'];m=mem['candidate_mask']&mem['history_mask'].unsqueeze(-1);lp=mem['log_prior'].float()
        roots+=int(m.sum());histories+=int(m.any(-1).sum());entropy+=float((-(lp.exp()*lp)*m).sum());nonfinite+=int((~torch.isfinite(out['logits'])).sum())
    model.train();nll=loss/tokens
    return dict(val_nll=nll,val_perplexity=math.exp(nll),val_target_tokens=int(tokens),root_candidate_count=roots/max(histories,1),candidate_entropy=entropy/max(histories,1),nonfinite_logits=nonfinite)


def worker(a):
    device=torch.device('cuda');cfg=load_config(a.config);cfg['data']['cache_dir']=a.cache_dir;cfg['seed']=TRAIN_SEED
    impl,backend,mode=ARMS[a.arm]['implementation'],ARMS[a.arm]['backend'],ARMS[a.arm]['mode']
    cfg['model'].update(forest_implementation=impl,search_backend=backend,reconstruction_mode=mode)
    cfg['train'].update(effective_tokens=a.effective_tokens,micro_batch_size=a.micro_batch_size,budget_tokens=a.steps*a.effective_tokens,output_dir=str(Path(a.output_dir)/a.arm))
    policy=PrecisionPolicy(**cfg['precision']);policy.configure_backend();torch.set_num_threads(4);set_seed(TRAIN_SEED)
    if 'compile' in a.arm:configure_inductor()
    train=StoryDataset(a.cache_dir);dev=StoryDataset(a.cache_dir,'dev')
    loader=DataLoader(dev,batch_size=8,shuffle=False,collate_fn=Collator(),num_workers=0,generator=torch.Generator().manual_seed(0))
    val_batches=[];
    for i,b in enumerate(loader):
        if i>=a.eval_batches:break
        val_batches.append(to_device(b,device))
    model=configure_model(LatentForestLM(cfg['model'],TRAIN_SEED).to(device),policy);init_hash=weight_hash(model)
    optimizer=make_optimizer(model,cfg['train']);schedule=make_schedule(optimizer,cfg['train']);sampler=StatefulSampler(train,TRAIN_SEED)
    root=Path(cfg['train']['output_dir']);root.mkdir(parents=True,exist_ok=True)
    atomic_json(dict(config=cfg,arm=a.arm,init_weight_sha256=init_hash,data_fingerprint=train.manifest['fingerprint'],environment=device_info(),backend=backend_state()),root/'run.json')
    rows=[];total_train=0.;nonfinite_steps=0;tick0=time.perf_counter();val=validate(model,val_batches,policy,device)
    rows.append(dict(step=0,target_tokens=0,**val,train_seconds=0.,elapsed_seconds=0.));print(json.dumps(rows[-1]),flush=True)
    for step in range(1,a.steps+1):
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();tick=time.perf_counter()
        metrics=train_update(model,sampler,optimizer,schedule,a.effective_tokens,a.micro_batch_size,device,policy.outer_autocast,cfg['train']['grad_clip'])
        torch.cuda.synchronize();seconds=time.perf_counter()-tick;total_train+=seconds
        if not math.isfinite(metrics['grad_norm']):nonfinite_steps+=1
        row=dict(step=step,target_tokens=step*a.effective_tokens,**metrics,step_seconds=seconds,tokens_per_second=metrics['step_tokens']/seconds,train_seconds=total_train,elapsed_seconds=time.perf_counter()-tick0,
                 peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30,cache_misses=model.forest._geometry_cache.misses if hasattr(model.forest,'_geometry_cache') else None,
                 cache_build_ms=model.forest._geometry_cache.build_host_ms if hasattr(model.forest,'_geometry_cache') else None)
        if step%a.eval_every==0 or step==a.steps:row.update(validate(model,val_batches,policy,device))
        if 'compile' in a.arm:c=compile_counters();row['compiled_frames']=int(c.get('stats',{}).get('unique_graphs',0));row['fx_graphs_compiled']=int(c.get('inductor',{}).get('fxgraph_cache_miss',0))  # the 'frames' counter is absent in torch 2.12 (the Stage E runs recorded 0 here)
        rows.append(row);print(json.dumps(row),flush=True)
        pd.DataFrame(rows).to_csv(root/'metrics.csv',index=False)
    save_checkpoint(root/'final.pt',model,optimizer,schedule,sampler,cfg,dict(step=a.steps,target_tokens=a.steps*a.effective_tokens,train_seconds=total_train,elapsed_seconds=time.perf_counter()-tick0,last_gradients={}),train.manifest['fingerprint'])
    summary=dict(arm=a.arm,implementation=impl,search_backend=backend,reconstruction_mode=mode,steps=a.steps,effective_tokens_per_update=a.effective_tokens,micro_batch_size=a.micro_batch_size,
        ce_target_tokens=a.steps*a.effective_tokens,train_wall_seconds=total_train,total_wall_seconds=time.perf_counter()-tick0,tokens_per_second_train_wall=a.steps*a.effective_tokens/total_train,
        final_train_nll_last10=sum(r['train_nll'] for r in rows[-10:])/10,final_val_nll=rows[-1]['val_nll'],initial_val_nll=rows[0]['val_nll'],nonfinite_gradient_steps=nonfinite_steps,
        mean_grad_norm=sum(r['grad_norm'] for r in rows[1:])/a.steps,mean_span_scorer_grad_norm=sum(r['span_scorer_grad_norm'] for r in rows[1:])/a.steps,
        final_candidate_entropy=rows[-1]['candidate_entropy'],final_root_candidate_count=rows[-1]['root_candidate_count'],init_weight_sha256=init_hash,peak_gpu_gib=max(r['peak_gpu_gib'] for r in rows[1:]),
        cache_misses=rows[-1]['cache_misses'],cache_build_seconds=(rows[-1]['cache_build_ms'] or 0)/1000,
        timing_scope='train_wall = synchronized wall time of train_update (sampling, collation, H2D, forward, backward, clip, AdamW, CPU scheduling and GPU waits); excludes validation and checkpoint IO; not pure GPU kernel time')
    atomic_json(summary,root/'summary.json');print(json.dumps(summary),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='configs/v35/baseline.yaml');p.add_argument('--cache-dir',default='cache/tinystories')
    p.add_argument('--output-dir',default='reports/cky_v35/training');p.add_argument('--arms',nargs='+',choices=list(ARMS),default=['v3','v3c'])
    p.add_argument('--steps',type=int,default=150);p.add_argument('--effective-tokens',type=int,default=4096);p.add_argument('--micro-batch-size',type=int,default=16)
    p.add_argument('--eval-every',type=int,default=25);p.add_argument('--eval-batches',type=int,default=50);p.add_argument('--worker',action='store_true');p.add_argument('--arm')
    a=p.parse_args()
    if a.worker:return worker(a)
    root=Path(a.output_dir);(root/'logs').mkdir(parents=True,exist_ok=True)
    for arm in a.arms:
        cmd=[sys.executable,__file__,'--worker','--arm',arm,'--config',a.config,'--cache-dir',a.cache_dir,'--output-dir',str(root),'--steps',str(a.steps),'--effective-tokens',str(a.effective_tokens),
             '--micro-batch-size',str(a.micro_batch_size),'--eval-every',str(a.eval_every),'--eval-batches',str(a.eval_batches)]
        print('training arm',arm,flush=True)
        with (root/'logs'/f'train_{arm}.log').open('w',encoding='utf-8') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
        if r.returncode:print('ARM FAILED (log retained):',arm,flush=True)
    summaries=[json.loads((root/arm/'summary.json').read_text(encoding='utf-8')) for arm in a.arms if (root/arm/'summary.json').exists()]
    pd.DataFrame(summaries).to_csv(root/'training_throughput.csv',index=False)
    curves=[]
    for arm in a.arms:
        if (root/arm/'metrics.csv').exists():
            m=pd.read_csv(root/arm/'metrics.csv');m['arm']=arm;curves.append(m)
    if curves:pd.concat(curves,ignore_index=True).to_csv(root/'training_curves.csv',index=False)
    print(pd.DataFrame(summaries).to_string() if summaries else 'no summaries')

if __name__=='__main__':main()
