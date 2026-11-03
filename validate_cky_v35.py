"""V3.5 correctness: V3-C search/gradient identity, C1 vs C2 agreement, compile-backend identity.

All comparisons reuse the V3 protocol (v3_study.forest_pair / compare_search /
global_gradients) with the ORIGINAL strict tolerances: FP32 atol 3e-6 / rtol 3e-4,
BF16 atol 1.5e-4 / rtol 0.04. Failures are recorded, never relaxed.
"""
from __future__ import annotations
import argparse,gc,itertools,json,traceback
from pathlib import Path
import torch
from benchmark_cky_v2 import cpu_snapshot
from models.composer import Composer
from models.forest import SemanticForest
from models.forest_tensorized import SCORE_RECONSTRUCT
from models.model import LatentForestLM
from models.precision_policy import POLICIES,configure_model
from reader_precision_study import tensor_errors,global_gradients
from v3_study import forest_pair,model_pair,compare_search,assert_rng_equal
from validate_cky_v25 import small_config,model_batch,gradient_statistics
from utils import atomic_json,load_config,rng_state,restore_rng,set_seed

ARMS={'v3':('lazy_tensorized','eager','dag'),'v3c':(SCORE_RECONSTRUCT,'eager','dag'),'v3c_tree':(SCORE_RECONSTRUCT,'eager','tree'),
      'v3_function':('lazy_tensorized','function','dag'),'v3_compile_default':('lazy_tensorized','compile_default','dag'),'v3_compile_ro':('lazy_tensorized','compile_reduce_overhead','dag'),
      'v3c_compile_default':(SCORE_RECONSTRUCT,'compile_default','dag'),'v3c_compile_ro':(SCORE_RECONSTRUCT,'compile_reduce_overhead','dag')}


def set_arm(forest,arm):
    impl,backend,mode=ARMS[arm];forest.implementation=impl;forest.search_backend=backend;forest.reconstruction_mode=mode


def span_diff(a,b):
    """First differing spans by width: candidate identities and score deltas (for compile mismatch tracing)."""
    rows=[]
    for span in sorted(a['spans'],key=lambda s:(s[1]-s[0],s[0])):
        x,y=a['spans'][span],b['spans'][span]
        for key in ('selected_flat','retained_flat','candidate_mask','scores'):
            if x[key] is None:continue
            u,v=x[key].detach().cpu(),y[key].detach().cpu()
            if u.shape!=v.shape or not torch.equal(u,v):
                where=(u!=v).nonzero()[:4].tolist() if u.shape==v.shape else []
                rows.append(dict(span=list(span),width=span[1]-span[0],field=key,mismatches=int((u!=v).sum()) if u.shape==v.shape else -1,first_positions=where,
                    reference=[u[tuple(i)].item() for i in where],other=[v[tuple(i)].item() for i in where],
                    max_abs_score_delta=float((u.float()-v.float()).abs().max()) if key=='scores' and u.shape==v.shape else None))
        if len(rows)>=12:break
    return rows


def forest_arm_pair(batch,words,k,r,vector,device,precision,policy,arm_a,arm_b,grad=True):
    """Fixed-input forest comparison between two arms: search identity, RNG, roots, input gradients."""
    set_seed(1000+batch*100+words*10+k+r);torch.set_num_threads(1);pp=POLICIES['B' if precision=='bf16' else 'E']
    if device=='cuda':pp.configure_backend()
    n=batch*2;dim=8;c=3 if vector else 0
    f=SemanticForest(Composer(dim,16),k,r,8.,1. if policy=='explore' else .0,'lazy_tensorized').to(device);f.precision_policy=pp;f.train(policy=='explore')
    leaves=torch.randn(n,words,dim,device=device,requires_grad=True)
    scores=(torch.randn(n,words,words,device=device)*.15);coeff=torch.randn(n,words,words,c,device=device)*.1
    if precision=='bf16':scores=scores.bfloat16();coeff=coeff.bfloat16()
    scores.requires_grad_();coeff.requires_grad_()
    lengths=([words,max(1,words-1),words//2,1,0,words,2,3]*2)[:n]
    mask=torch.arange(words,device=device)[None,:]<torch.tensor([min(words,x) for x in lengths],device=device)[:,None]
    if policy in ('ties','near_ties'):
        with torch.no_grad():
            scores.zero_()
            if policy=='near_ties':scores[:,0,-1]=torch.finfo(scores.dtype).eps
    elif policy=='prune':f.delta=0.
    elif policy=='empty':mask.zero_()
    elif policy=='all_inf':
        with torch.no_grad():scores.fill_(-torch.inf)
    initial=rng_state();outs=[];ends=[];grads=[];work=[]
    weights=None
    for arm in (arm_a,arm_b):
        restore_rng(initial);set_arm(f,arm);f.zero_grad(set_to_none=True)
        for t in (leaves,scores,coeff):t.grad=None
        with pp.outer_context(device):out=f(leaves,scores,coeff,mask,return_spans=True,return_trees=True)
        ends.append(rng_state())
        if grad:
            if weights is None:weights=torch.randn_like(out['log_prior'])
            with pp.outer_context(device):
                total=(out['log_prior'].float()*weights).sum()+out['semantic'].float().square().sum()+out['vector_coeff'].float().square().sum()
            total.backward()
            grads.append({name:None if t.grad is None else t.grad.detach().cpu().clone() for name,t in [('leaves',leaves),('scores',scores),('coeff',coeff)]+list(f.named_parameters())})
        outs.append(cpu_snapshot(out));work.append(dict(f.last_workload));del out
    report=dict(batch=batch,words=words,k=k,r=r,vector=vector,device=device,precision=precision,policy=policy,arms=[arm_a,arm_b],status='passed')
    try:assert_rng_equal(*ends)
    except AssertionError:report['status']='failed';report['rng_equal']=False
    try:compare_search(outs[0],outs[1],precision=='bf16')
    except AssertionError as exc:report['status']='failed';report['search_error']=str(exc)[:400];report['span_diff']=span_diff(outs[0],outs[1])
    if grad:
        atol,rtol=(1.5e-4,.04) if precision=='bf16' else (3e-6,3e-4);failures=[]
        for name in grads[0]:
            x,y=grads[0][name],grads[1][name]
            if (x is None)!=(y is None):failures.append(name);continue
            if x is None:continue
            if not torch.allclose(x.float(),y.float(),atol=atol,rtol=rtol):failures.append(name)
        report['gradient']=gradient_statistics(grads[0],grads[1]);report['gradient_strict_failures']=failures
        report['score_gradient_norm']=[float(g['scores'].float().norm()) if g['scores'] is not None else None for g in grads]
        if failures:report['status']='failed'
    report['composer_rows']=[w.get('v25_actual_composer_rows') for w in work];report['reconstruction_nodes']=work[1].get('score_reconstruction_nodes')
    return report


def lm_arm_pair(device,precision,arm_a,arm_b,main=False,batch=2,mode='forest_vector4'):
    set_seed(42);torch.set_num_threads(1);pp=POLICIES['B' if precision=='bf16' else 'E'];pp.configure_backend()
    cfg=load_config('configs/main_125m.yaml') if main else small_config(mode);cfg['model']['mode']=mode
    model=configure_model(LatentForestLM(cfg['model']).to(device).train(),pp);model.forest.capture_spans=True
    inputs=model_batch(cfg,batch=batch,words=24 if main else 8,histories=4 if main else 2,target_length=16 if main else 8,device=device)
    initial=rng_state();outputs=[];grads=[];ends=[]
    for arm in (arm_a,arm_b):
        set_arm(model.forest,arm);model.zero_grad(set_to_none=True);restore_rng(initial)
        with pp.outer_context(device):out=model(**inputs)
        ends.append(rng_state());outputs.append(cpu_snapshot(out));out['loss'].backward()
        grads.append({name:None if p.grad is None else p.grad.detach().cpu().clone() for name,p in model.named_parameters()});del out
    report=dict(device=device,precision=precision,arms=[arm_a,arm_b],main=main,batch=batch,mode=mode,status='passed')
    try:assert_rng_equal(*ends)
    except AssertionError:report['status']='failed';report['rng_equal']=False
    try:compare_search(outputs[0]['memory'],outputs[1]['memory'],precision=='bf16')
    except AssertionError as exc:report['status']='failed';report['search_error']=str(exc)[:400];report['span_diff']=span_diff(outputs[0]['memory'],outputs[1]['memory'])
    report['logits_bitwise']=bool(torch.equal(outputs[0]['logits'],outputs[1]['logits']));report['loss_bitwise']=bool(torch.equal(outputs[0]['loss'],outputs[1]['loss']))
    report['forward']={k:tensor_errors(outputs[0][k],outputs[1][k]) for k in ('logits','loss')}
    report['root']={k:tensor_errors(outputs[0]['memory'][k],outputs[1]['memory'][k]) for k in ('semantic','scores','vector_coeff','log_prior')}
    g=global_gradients(grads[0],grads[1],strict_label='bf16' if precision=='bf16' else 'fp32');report['gradient']=g
    assert g['modules']['span_scorer']['gradient_norm_lazy']>0
    if precision=='fp32' and g['strict_failed_parameter_count']:report['status']='failed'
    if not report['logits_bitwise']:report['status']='failed'
    del model,grads,outputs;gc.collect()
    if device=='cuda':torch.cuda.empty_cache()
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',default='reports/cky_v35')
    p.add_argument('--phases',nargs='+',choices=['matrix','gradients','tree','compile'],default=['matrix','gradients','tree','compile']);p.add_argument('--cpu-only',action='store_true');p.add_argument('--quick',action='store_true')
    a=p.parse_args();root=Path(a.output_dir);root.mkdir(parents=True,exist_ok=True)
    devices=[('cpu','fp32')]+([] if a.cpu_only or not torch.cuda.is_available() else [('cuda','fp32'),('cuda','bf16')])
    try:
        if 'matrix' in a.phases:
            correctness=dict(implementation=SCORE_RECONSTRUCT,reference='lazy_semantic',matrix=[],edge_cases=[],status='running')
            grid=itertools.product([1,2,4],[1,4,8,16,24],[1,2,4],[4,12],[False,True],['normal','explore'])
            if a.quick:grid=itertools.product([2],[1,8,24],[4],[12],[True],['normal','explore'])
            grid=list(grid)
            for device,precision in devices:
                for b,w,k,r,v,policy in grid:
                    correctness['matrix'].append(forest_pair(b,w,k,r,v,device,precision,policy,SCORE_RECONSTRUCT))
                    if len(correctness['matrix'])%60==0:print('matrix',len(correctness['matrix']),device,precision,flush=True);atomic_json(correctness,root/'correctness.json')
                for policy,v in itertools.product(['ties','near_ties','prune','empty','all_inf','partial_inf'],[False,True]):
                    correctness['edge_cases'].append(forest_pair(2,8,4,12,v,device,precision,policy,SCORE_RECONSTRUCT))
            correctness['status']='passed';atomic_json(correctness,root/'correctness.json');print('matrix passed',len(correctness['matrix']),flush=True)
        if 'gradients' in a.phases:
            gradients=dict(reference='lazy_semantic',implementation=SCORE_RECONSTRUCT,cases=[],fp32_tolerance={'atol':3e-6,'rtol':.0003},bf16_tolerance={'atol':.00015,'rtol':.04})
            for device,precision in devices:
                for mode in ('forest_scalar4','forest_vector4'):
                    z=model_pair(mode,device,precision,implementation=SCORE_RECONSTRUCT);gradients['cases'].append(z);atomic_json(gradients,root/'gradients.json')
                    print('lm pair',device,precision,mode,'strict failures',z['gradient']['strict_failed_parameter_count'],flush=True)
                    if precision=='fp32':assert z['gradient']['strict_failed_parameter_count']==0,z['gradient']['strict_failed_parameters']
            if not a.cpu_only and torch.cuda.is_available():
                for precision,batch in itertools.product(['fp32','bf16'],[1,4]):
                    z=model_pair(device='cuda',precision=precision,main=True,batch=batch,implementation=SCORE_RECONSTRUCT);gradients['cases'].append(z);atomic_json(gradients,root/'gradients.json')
                    print('main',precision,batch,'strict failures',z['gradient']['strict_failed_parameter_count'],flush=True)
                    if precision=='fp32':assert z['gradient']['strict_failed_parameter_count']==0,z['gradient']['strict_failed_parameters']
            # Direct V3 -> V3-C pairs including the full-LM gradient (same reference arm as the benchmark).
            gradients['v3_vs_v3c']=[lm_arm_pair(device,precision,'v3','v3c',main=False) for device,precision in devices]
            if not a.cpu_only and torch.cuda.is_available():gradients['v3_vs_v3c']+=[lm_arm_pair('cuda',precision,'v3','v3c',main=True,batch=b) for precision,b in itertools.product(['fp32','bf16'],[1,4])]
            gradients['status']='complete; BF16 strict mismatches preserved';atomic_json(gradients,root/'gradients.json')
        if 'tree' in a.phases:
            tree=dict(description='C1 shared-DAG vs C2 per-root-tree reconstruction: forward bitwise, gradients strict',cases=[])
            for device,precision in devices:
                for b,w,policy in [(2,8,'normal'),(2,24,'explore'),(4,16,'normal'),(1,1,'normal'),(2,8,'empty')]:
                    tree['cases'].append(forest_arm_pair(b,w,4,12,True,device,precision,policy,'v3c','v3c_tree'))
            tree['failed']=[c for c in tree['cases'] if c['status']!='passed'];atomic_json(tree,root/'tree_vs_dag.json');print('tree vs dag failures',len(tree['failed']),flush=True)
        if 'compile' in a.phases and torch.cuda.is_available() and not a.cpu_only:
            from models.cky_compile import reset_compile_state,compile_counters
            compile=dict(description='Search identity of function/compiled width step against eager V3 (same weights, RNG); fixed-shape synthetic forests and full LM.',arms={})
            for arm in ('v3_function','v3_compile_default','v3_compile_ro','v3c_compile_default','v3c_compile_ro'):
                reset_compile_state();cases=[]
                for precision in ('fp32','bf16'):
                    for b,w,policy in [(2,8,'normal'),(2,24,'explore'),(4,24,'normal'),(2,8,'ties'),(2,8,'near_ties')]:
                        cases.append(forest_arm_pair(b,w,4,12,True,'cuda',precision,policy,'v3' if not arm.startswith('v3c') else 'v3c',arm))
                        print(arm,precision,b,w,policy,cases[-1]['status'],flush=True)
                lm=[lm_arm_pair('cuda',precision,'v3' if not arm.startswith('v3c') else 'v3c',arm,main=False) for precision in ('fp32','bf16')]
                lm.append(lm_arm_pair('cuda','bf16','v3' if not arm.startswith('v3c') else 'v3c',arm,main=True,batch=4))
                compile['arms'][arm]=dict(forest_cases=cases,lm_cases=lm,counters=compile_counters(),failed=[c for c in cases+lm if c['status']!='passed'])
                atomic_json(compile,root/'compile_identity.json')
            # Negative control: without emulate_precision_casts the width-2 BF16 leaf addition is computed in FP32.
            import torch._inductor.config as ic
            reset_compile_state();ic.emulate_precision_casts=False;ic.fallback_random=True
            import models.cky_compile as cc
            cc._compiled['compile_default']=torch.compile(cc.width_step,mode='default',dynamic=False,fullgraph=True)
            try:
                control=[forest_arm_pair(2,w,4,12,True,'cuda','bf16','normal','v3','v3_compile_default',grad=False) for w in (8,24)]
            finally:
                reset_compile_state()
            compile['negative_control_no_emulate_precision_casts']=control;atomic_json(compile,root/'compile_identity.json')
    except Exception as exc:
        atomic_json(dict(error=repr(exc),traceback=traceback.format_exc()),root/'validation_failure.json');raise
    print('V3.5 validation completed',flush=True)

if __name__=='__main__':main()
