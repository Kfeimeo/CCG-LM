"""Vectorized geometry builder must be bitwise identical to the original Python builder."""
import pytest
import torch
from models.cky_index_cache import CKYIndexCache


def same_plan(a,b):
    assert a.slots==b.slots and a.layouts==b.layouts and a.size==b.size and a.bank_size==b.bank_size and a.root_slots==b.root_slots
    assert torch.equal(a.root_gather,b.root_gather) and torch.equal(a.id_bank,b.id_bank)
    assert len(a.widths)==len(b.widths) and a.widths[0] is None and b.widths[0] is None
    for x,y in zip(a.widths[1:],b.widths[1:]):
        for f in ('width','spans','candidates','budget','keep','metadata','bases'):assert getattr(x,f)==getattr(y,f),f
        for f in ('left','right','parent','decode','ends'):assert torch.equal(getattr(x,f),getattr(y,f)),f
        for f in ('left_ids','right_ids'):
            u,v=getattr(x,f),getattr(y,f);assert (u is None)==(v is None)
            if u is not None:assert torch.equal(u,v),f
    assert a.device_bytes==b.device_bytes


@pytest.mark.parametrize('words,batch,k,r',[(1,1,4,12),(2,3,4,12),(5,2,4,12),(8,7,2,4),(12,16,1,4),(24,4,4,12),(24,13,4,12)])
@pytest.mark.parametrize('full_decode',[False,True])
def test_fast_build_matches_reference(words,batch,k,r,full_decode):
    CKYIndexCache._unit_geometry.clear()
    a=CKYIndexCache._build_reference(words,batch,k,r,torch.device('cpu'),full_decode)
    b=CKYIndexCache._build_fast(words,batch,k,r,torch.device('cpu'),full_decode)
    same_plan(a,b)
    same_plan(a,CKYIndexCache._build_fast(words,batch,k,r,torch.device('cpu'),full_decode))  # unit geometry reuse


def test_unbounded_and_cache_path():
    CKYIndexCache._unit_geometry.clear()
    a=CKYIndexCache._build_reference(6,2,None,None,torch.device('cpu'),True);b=CKYIndexCache._build_fast(6,2,None,None,torch.device('cpu'),True);same_plan(a,b)
    c=CKYIndexCache(4);plan,hit,_=c.get(6,2,None,None,'cpu',True);assert not hit;same_plan(a,plan)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_fast_build_cuda_matches_reference():
    CKYIndexCache._unit_geometry.clear();d=torch.device('cuda')
    for words,batch in [(24,16),(24,100),(13,4)]:
        same_plan(CKYIndexCache._build_reference(words,batch,4,12,d,True),CKYIndexCache._build_fast(words,batch,4,12,d,True))
