"""Exact search, RNG and full-gradient checks shared by V3 experiments."""
from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))  # repo root: run from the repo root or via python -m
del _sys, _Path
import gc
import numpy as np
import torch
from models.forest import SemanticForest
from models.composer import Composer
from models.model import LatentForestLM
from models.precision_policy import configure_model,POLICIES
from legacy.reader_v25.reader_precision_study import tensor_errors,global_gradients
from legacy.cky_v2.benchmark_cky_v2 import cpu_snapshot
from legacy.cky_v25.validate_cky_v25 import compare_forests,small_config,model_batch,gradient_statistics
from utils import rng_state,restore_rng,set_seed,load_config


def assert_rng_equal(a,b):
    assert a['python']==b['python']
    assert a['numpy'][0]==b['numpy'][0] and np.array_equal(a['numpy'][1],b['numpy'][1]) and a['numpy'][2:]==b['numpy'][2:]
    assert torch.equal(a['torch'],b['torch'])
    assert len(a['cuda'])==len(b['cuda']) and all(torch.equal(x,y) for x,y in zip(a['cuda'],b['cuda']))


def compare_search(a,b,bf16=False):
    # Existing strict V2.5 thresholds, including bitwise structure scores/masks.
    metrics=compare_forests(a,b,bf16)
    assert set(a['spans'])==set(b['spans'])
    for span in a['spans']:
        x,y=a['spans'][span],b['spans'][span]
        assert x['candidate_base']==y['candidate_base'],('base',span)
        assert x['metadata']==y['metadata'],('metadata',span)
        for key in ('left_id','right_id'):
            if key in x:assert torch.equal(x[key],y[key]),(key,span)
    for key in ('semantic','vector_coeff','scores','log_prior','candidate_mask'):
        assert torch.equal(a[key],b[key]),('same DAG must emit bitwise equal roots',key)
    if 'trees' in a:assert a['trees']==b['trees']
    return metrics


def forest_pair(batch,words,k,r,vector,device='cpu',precision='fp32',policy='normal',implementation='lazy_tensorized_a'):
    set_seed(1000+batch*100+words*10+k+r);torch.set_num_threads(1)
    pp=POLICIES['B' if precision=='bf16' else 'E']
    if device=='cuda':pp.configure_backend()
    n=batch*2;dim=8;c=3 if vector else 0
    f=SemanticForest(Composer(dim,16),k,r,8.,1. if policy=='explore' else .0,'lazy_semantic').to(device)
    f.precision_policy=pp;f.train(policy=='explore')
    leaves=torch.randn(n,words,dim,device=device)
    scores=torch.randn(n,words,words,device=device)*.15
    # Reproduce actual Scorer output storage under outer AMP, not just its context.
    if precision=='bf16':scores=scores.bfloat16()
    coeff=torch.randn(n,words,words,c,device=device)*.1
    if precision=='bf16':coeff=coeff.bfloat16()
    lengths=([words,max(1,words-1),words//2,1,0,words,2,3]*2)[:n]
    mask=torch.arange(words,device=device)[None,:]<torch.tensor([min(words,x) for x in lengths],device=device)[:,None]
    if policy in ('ties','near_ties'):
        scores.zero_()
        if policy=='near_ties':scores[:,0,-1]=torch.finfo(scores.dtype).eps
    elif policy=='prune':f.delta=0.
    elif policy=='empty':mask.zero_()
    elif policy=='all_inf':scores.fill_(-torch.inf)
    elif policy=='partial_inf' and words>1:scores[:,0,1]=-torch.inf
    initial=rng_state();outputs=[];ends=[];work=[]
    for impl in ('lazy_semantic',implementation):
        restore_rng(initial);f.implementation=impl
        with torch.no_grad(),pp.outer_context(device):out=f(leaves,scores,coeff,mask,return_spans=True,return_trees=True)
        outputs.append(out);ends.append(rng_state());work.append(f.last_workload.copy())
    assert_rng_equal(*ends);compare_search(*outputs,precision=='bf16')
    assert work[0]['v25_actual_composer_rows']==work[1]['v25_actual_composer_rows']
    assert work[0]['v25_composer_calls']==work[1]['v25_composer_calls']
    return dict(batch=batch,words=words,k=k,r=r,vector=vector,device=device,precision=precision,policy=policy,
                implementation=implementation,spans=len(outputs[0]['spans']),rng_equal=True,roots_bitwise_equal=True,
                composer_rows=work[1]['v25_actual_composer_rows'],status='passed')


def model_pair(mode='forest_vector4',device='cpu',precision='fp32',main=False,batch=2,implementation='lazy_tensorized_a'):
    set_seed(42);torch.set_num_threads(1);pp=POLICIES['B' if precision=='bf16' else 'E'];pp.configure_backend()
    cfg=load_config('configs/main_125m.yaml') if main else small_config(mode)
    cfg['model']['mode']=mode
    model=configure_model(LatentForestLM(cfg['model']).to(device).train(),pp);model.forest.capture_spans=True
    inputs=model_batch(cfg,batch=batch,words=24 if main else 8,histories=4 if main else 2,target_length=16 if main else 8,device=device)
    initial=rng_state();outputs=[];grads=[];ends=[];reader_values=[]
    for impl in ('lazy_semantic',implementation):
        model.forest.implementation=impl;model.zero_grad(set_to_none=True);restore_rng(initial)
        rv={};handles=[r.register_forward_hook(lambda m,i,o,name=name:rv.__setitem__(name,o[0].detach().cpu().clone())) for name,r in model.readers.items()]
        with pp.outer_context(device):out=model(**inputs)
        for handle in handles:handle.remove()
        reader_values.append(rv);ends.append(rng_state())
        outputs.append(cpu_snapshot(out));out['loss'].backward()
        grads.append({name:None if p.grad is None else p.grad.detach().cpu().clone() for name,p in model.named_parameters()})
        del out
    assert_rng_equal(*ends);compare_search(outputs[0]['memory'],outputs[1]['memory'],precision=='bf16')
    forward={key:tensor_errors(outputs[0][key],outputs[1][key]) for key in ('logits','loss')}
    forward['root']={key:tensor_errors(outputs[0]['memory'][key],outputs[1]['memory'][key]) for key in ('semantic','scores','vector_coeff','log_prior')}
    forward['readers']={name:tensor_errors(reader_values[0][name],reader_values[1][name]) for name in reader_values[0]}
    assert torch.equal(outputs[0]['logits'],outputs[1]['logits']) and torch.equal(outputs[0]['loss'],outputs[1]['loss'])
    gradient=global_gradients(*grads,strict_label='bf16' if precision=='bf16' else 'fp32')
    assert all(v['nan']+v['inf']==0 for v in gradient['nonfinite_gradients'].values())
    assert gradient['modules']['span_scorer']['gradient_norm_lazy']>0
    gradient['parameters']={name:gradient_statistics({name:g},{name:grads[1][name]}) for name,g in grads[0].items()}
    answer=dict(mode=mode,device=device,precision=precision,main=main,batch=batch,implementation=implementation,forward=forward,gradient=gradient,rng_equal=True)
    del model,grads,outputs;gc.collect()
    if device=='cuda':torch.cuda.empty_cache()
    return answer


def assert_state_equal(a,b):
    if isinstance(a,torch.Tensor):assert isinstance(b,torch.Tensor) and torch.equal(a.cpu(),b.cpu())
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for key in a:assert_state_equal(a[key],b[key])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):assert_state_equal(x,y)
    else:assert a==b
