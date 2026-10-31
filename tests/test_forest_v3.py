"""Final V3 identities, sparse reachability and checkpoint migration/replay."""
import copy
import numpy as np
import pytest
import torch
from models.forest import SemanticForest
from models.composer import Composer
from models.model import LatentForestLM
from models.precision_policy import POLICIES,configure_model
from v3_study import forest_pair,model_pair,compare_search,assert_rng_equal
from utils import rng_state,restore_rng,set_seed

@pytest.mark.parametrize('words,k,r',[(1,1,4),(4,4,12),(8,2,4),(16,4,12),(24,4,12)])
@pytest.mark.parametrize('policy',['normal','explore','ties','near_ties','prune','empty','partial_inf','all_inf'])
def test_final_search(words,k,r,policy):
    forest_pair(2,words,k,r,True,policy=policy,implementation='lazy_tensorized')

@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
@pytest.mark.parametrize('precision',['fp32','bf16'])
@pytest.mark.parametrize('policy',['normal','explore','ties','near_ties','all_inf'])
def test_final_cuda(precision,policy):
    forest_pair(4,24,4,12,True,'cuda',precision,policy,'lazy_tensorized')

@pytest.mark.parametrize('mode',['forest_scalar4','forest_vector4'])
def test_final_full_gradients(mode):
    assert model_pair(mode,implementation='lazy_tensorized')['gradient']['strict_failed_parameter_count']==0

@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
@pytest.mark.parametrize('precision',['fp32','bf16'])
def test_final_cuda_full_gradients(precision):
    z=model_pair(device='cuda',precision=precision,implementation='lazy_tensorized')
    if precision=='fp32':assert z['gradient']['strict_failed_parameter_count']==0

@pytest.mark.parametrize('no_history',[False,True])
def test_invalid_history_and_state_dict(tiny_config,batch,no_history):
    cfg=copy.deepcopy(tiny_config['model']);model=LatentForestLM(cfg).eval();model.forest.capture_spans=True
    inputs=dict(batch)
    if no_history:
        inputs['history_ids']=batch['history_ids'][:,:0];inputs['word_mask']=batch['word_mask'][:,:0];inputs['history_mask']=batch['history_mask'][:,:0]
    else:inputs['history_mask']=torch.zeros_like(batch['history_mask'])
    keys=list(model.state_dict());initial=rng_state();outputs=[]
    for impl in ('lazy_semantic','lazy_tensorized'):
        model.forest.implementation=impl;restore_rng(initial);outputs.append(model(**inputs))
    assert torch.equal(outputs[0]['logits'],outputs[1]['logits'])
    assert not outputs[1]['memory']['candidate_mask'].any()
    assert keys==list(model.state_dict())
    assert not any('cache' in k for k in keys)

@pytest.mark.parametrize('empty',[False,True])
def test_shared_closure_gradient_and_exact_unbounded(empty):
    f=SemanticForest(Composer(8,16),k=None,max_compositions=None,delta=None,exploration=0).train()
    h=torch.randn(2,5,8,requires_grad=True);s=torch.randn(2,5,5,requires_grad=True);u=torch.randn(2,5,5,3,requires_grad=True)
    m=torch.ones(2,5,dtype=torch.bool)
    if empty:m.zero_()
    all_grads=[];outputs=[]
    for impl in ('lazy_semantic','lazy_tensorized'):
        f.implementation=impl;f.zero_grad(set_to_none=True)
        for t in (h,s,u):t.grad=None
        out=f(h,s,u,m,return_spans=True,return_trees=True)
        (out['semantic'].square().sum()+out['vector_coeff'].square().sum()+out['log_prior'].square().sum()).backward()
        all_grads.append([None if t.grad is None else t.grad.clone() for t in (h,s,u,*f.parameters())]);outputs.append(out)
    compare_search(*outputs)
    if not empty:assert len(outputs[1]['trees'][0])==14
    for a,b in zip(*all_grads):
        assert (a is None)==(b is None)
        if a is not None:torch.testing.assert_close(a,b,atol=3e-6,rtol=.0003)

class ReplayDataset:
    def __init__(self):
        self.examples=[dict(history=[[4,5,6,7],[5,6,7]],input_ids=[2,4,5,6,7],labels=[4,5,6,7,3],story_id=str(i),sample_id=i) for i in range(4)]
        self.bpe_offsets=np.array([0,5,10,15,20])
    def __len__(self):return 4
    def __getitem__(self,i):return copy.deepcopy(self.examples[i])

@pytest.mark.parametrize('legacy',[False,True])
def test_checkpoint_switch_and_replay(tiny_config,tmp_path,legacy):
    from train import make_optimizer,make_schedule,train_update,save_checkpoint,load_checkpoint
    from data.dataset import StatefulSampler
    set_seed(123,True);policy=POLICIES['E'];policy.configure_backend()
    cfg=copy.deepcopy(tiny_config);cfg['model'].update(forest_implementation='lazy_semantic' if legacy else 'lazy_tensorized',dropout=.1,exploration=.25)
    cfg['precision']=policy.state_dict();dataset=ReplayDataset()
    def create(config):
        m=configure_model(LatentForestLM(config['model']),policy);o=make_optimizer(m,config['train'])
        return m,o,make_schedule(o,config['train']),StatefulSampler(dataset,42)
    m,o,sch,sam=create(cfg);args=dict(tokens=16,micro_batch_size=1,device=torch.device('cpu'),precision='fp32',clip=1.)
    train_update(m,sam,o,sch,**args);path=tmp_path/'checkpoint.pt'
    save_checkpoint(path,m,o,sch,sam,cfg,dict(step=1,target_tokens=16),'fixture')
    saved=torch.load(path,weights_only=False);assert saved['forest_implementation']==cfg['model']['forest_implementation']
    if legacy:
        saved.pop('forest_implementation');torch.save(saved,path)
    new_cfg=copy.deepcopy(cfg);new_cfg['model']['forest_implementation']='lazy_tensorized'
    r,ro,rs,ra=create(new_cfg);loaded=load_checkpoint(path,r,ro,rs,ra,new_cfg,'fixture')
    assert loaded['state']['step']==1 and loaded['state']['target_tokens']==16
    assert sam.state_dict()==ra.state_dict() and sch.state_dict()==rs.state_dict()
    for x,y in zip(m.parameters(),r.parameters()):assert torch.equal(x,y)
    if legacy:
        from validate_cky_v25 import model_batch
        m.forest.capture_spans=r.forest.capture_spans=True
        inputs=model_batch(cfg);state=rng_state();x=m(**inputs);end=rng_state();restore_rng(state);y=r(**inputs)
        compare_search(x['memory'],y['memory']);assert torch.equal(x['logits'],y['logits']);assert_rng_equal(end,rng_state())
    else:
        state=rng_state();expected=train_update(m,sam,o,sch,**args);end=rng_state();restore_rng(state)
        actual=train_update(r,ra,ro,rs,**args);assert actual==expected;assert_rng_equal(end,rng_state())
        for x,y in zip(m.parameters(),r.parameters()):assert torch.equal(x,y)
        assert sch.state_dict()==rs.state_dict() and sam.state_dict()==ra.state_dict()


def test_legacy_tail_extraction_against_frozen_source():
    import importlib.util,sys
    from models.forest_score_first import SemanticForest as Base
    spec=importlib.util.spec_from_file_location('frozen_v25_lazy','reports/cky_v3/baseline/forest_lazy.py')
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    class Frozen(module.LazyForestMixin,Base):pass
    a=Frozen(Composer(8,16));a.implementation='lazy_semantic'
    b=SemanticForest(Composer(8,16),implementation='lazy_semantic');b.load_state_dict(a.state_dict())
    h=torch.randn(2,8,8,requires_grad=True);s=torch.randn(2,8,8,requires_grad=True);u=torch.randn(2,8,8,3,requires_grad=True)
    m=torch.ones(2,8,dtype=torch.bool);state=rng_state();outputs=[];gradients=[]
    for f in (a,b):
        for t in (h,s,u):t.grad=None
        restore_rng(state);out=f(h,s,u,m,return_spans=True,return_trees=True)
        (out['semantic'].square().mean()+out['vector_coeff'].square().mean()+out['log_prior'].square().mean()).backward()
        outputs.append(out);gradients.append([t.grad.clone() for t in (h,s,u,*f.parameters())])
    compare_search(*outputs)
    assert all(torch.equal(x,y) for x,y in zip(*gradients))


def test_index_cache_is_static_across_updates_and_masks():
    f=SemanticForest(Composer(8,16),exploration=0,implementation='lazy_tensorized')
    h=torch.randn(2,8,8);s=torch.randn(2,8,8,requires_grad=True);u=torch.randn(2,8,8,3);mask=torch.ones(2,8,dtype=torch.bool)
    optimizer=torch.optim.AdamW([s],lr=.01)
    x=f(h,s,u,mask);x['log_prior'][:,0].sum().backward();optimizer.step()
    cache=f._geometry_cache;misses=cache.misses
    mask[1,4:]=False
    a=f(h,s,u,mask,return_spans=True);assert cache.misses==misses and cache.hits>0
    f.implementation='lazy_semantic';b=f(h,s,u,mask,return_spans=True);compare_search(b,a)
    f.to(dtype=torch.float64);assert not cache.entries
