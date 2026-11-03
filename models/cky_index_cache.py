"""Shape-only ragged CKY geometry. Never stores scores, masks or neural graphs."""
from __future__ import annotations
from collections import Counter,OrderedDict
from dataclasses import dataclass,replace
import time
import torch

@dataclass(frozen=True)
class WidthIndex:
    width: int
    spans: int
    candidates: int
    budget: int
    keep: int
    metadata: tuple
    left: torch.Tensor
    right: torch.Tensor
    parent: torch.Tensor
    decode: torch.Tensor
    ends: torch.Tensor
    bases: tuple
    left_ids: torch.Tensor | None = None
    right_ids: torch.Tensor | None = None

@dataclass(frozen=True)
class ChartIndex:
    widths: tuple
    slots: tuple
    layouts: tuple
    size: int
    bank_size: int
    root_slots: int
    root_gather: torch.Tensor
    id_bank: torch.Tensor
    device_bytes: int

class CKYIndexCache:
    """Per-forest bounded cache, cleared when the module moves device/dtype."""
    layout_version='compact-ragged-width-v1'
    def __init__(self,max_entries=8):
        self.entries=OrderedDict();self.max_entries=max_entries
        self.hits=self.misses=self.evictions=0;self.build_host_ms=0.
        # Diagnostic counters only (V3.5 audit); never consulted by the search.
        self.key_counts=Counter();self.eviction_counts=Counter();self.build_log=[]
    def clear(self):self.entries.clear()
    @staticmethod
    def short_key(key):return dict(words=key[1],rows=key[2],k=key[3],r=key[4],device=key[5],full_decode=key[8])
    def get(self,words,batch,k,r,device,full_decode=False):
        device=torch.device(device)
        key=(self.layout_version,words,batch,k,r,device.type,device.index,torch.long,full_decode)
        self.key_counts[key]+=1
        if key in self.entries:
            self.hits+=1;self.entries.move_to_end(key);return self.entries[key],True,0.
        start=time.perf_counter();value=self._build(words,batch,k,r,device,full_decode)
        elapsed=(time.perf_counter()-start)*1000;self.misses+=1;self.build_host_ms+=elapsed
        self.build_log.append((key,elapsed,value.device_bytes))
        self.entries[key]=value
        while len(self.entries)>self.max_entries:
            evicted,_=self.entries.popitem(last=False);self.evictions+=1;self.eviction_counts[evicted]+=1
        return value,False,elapsed
    def statistics(self):
        total=self.hits+self.misses
        return dict(hits=self.hits,misses=self.misses,evictions=self.evictions,hit_rate=self.hits/max(1,total),
            build_host_ms_total=self.build_host_ms,resident_entries=len(self.entries),max_entries=self.max_entries,
            resident_bytes=sum(v.device_bytes for v in self.entries.values()),
            key_frequency=[dict(**self.short_key(k),requests=c) for k,c in self.key_counts.most_common()],
            evicted_keys=[dict(**self.short_key(k),evictions=c) for k,c in self.eviction_counts.most_common()],
            builds=[dict(**self.short_key(k),build_host_ms=e,bytes=b) for k,e,b in self.build_log])
    @staticmethod
    def _build(words,batch,k,r,device,full_decode=False):
        if CKYIndexCache.fast_build:return CKYIndexCache._build_fast(words,batch,k,r,device,full_decode)
        return CKYIndexCache._build_reference(words,batch,k,r,device,full_decode)

    # V3.5 Stage D: the row-independent geometry (pair tables) is planned once per
    # (words,k,r,device); row-dependent ID tables are tensor expressions of it.
    # Output is bitwise identical to _build_reference (tests/test_index_cache_fast.py).
    fast_build=True
    _unit_geometry={}

    @staticmethod
    def _unit(words,k,r,device):
        key=(words,k,r,device.type,device.index)
        cached=CKYIndexCache._unit_geometry.get(key)
        if cached is not None:return cached
        slots=[0,1];budgets=[0,0];counts=[0,0]
        for width in range(2,words+1):
            e=sum(slots[o]*slots[width-o] for o in range(1,width))
            budget=min(e,r) if r is not None else e
            slots.append(min(budget,k) if k is not None else budget);budgets.append(budget);counts.append(e)
        bank_bases={};unit_bases={};layouts=[];bank=cid=0
        for width in range(1,words+1):
            bank_bases[width]=bank
            for i in range(words-width+1):
                unit_bases[i,i+width]=cid;layouts.append((cid,i,i+width,slots[width]));cid+=slots[width]
            bank+=(words-width+1)*slots[width]
        def tensor(data):return torch.tensor(data,device=device,dtype=torch.long)
        widths=[None]
        for width in range(2,words+1):
            meta=tuple((o,l,rr) for o in range(1,width) for l in range(slots[o]) for rr in range(slots[width-o]))
            left=[];right=[];decode=[];spans=words-width+1
            for i in range(spans):
                left.append([bank_bases[o]+i*slots[o]+l for o,l,rr in meta])
                right.append([bank_bases[width-o]+(i+o)*slots[width-o]+rr for o,l,rr in meta])
                decode.append([(unit_bases[i,i+o],unit_bases[i+o,i+width],l,rr,slots[o],slots[width-o]) for o,l,rr in meta])
            widths.append(dict(width=width,spans=spans,candidates=counts[width],budget=budgets[width],keep=slots[width],meta=meta,
                               left=tensor(left),right=tensor(right),parent=tensor([i*words+i+width-1 for i in range(spans)]),
                               unit_decode=tensor(decode),ends=tensor([i+width for i in range(spans)]),unit_bases=tuple(unit_bases[i,i+width] for i in range(spans))))
        root_slots=max(slots);root=tensor([[bank_bases[w]+j if j<slots[w] else bank for j in range(root_slots)] for w in range(1,words+1)])
        # Candidate ID of (row 0, every bank position) and the slot count of its span, in bank order.
        unit_id=tensor([unit_bases[i,i+w] for w in range(1,words+1) for i in range(words-w+1) for j in range(slots[w])])
        slot_offset=tensor([j for w in range(1,words+1) for i in range(words-w+1) for j in range(slots[w])])
        slot_of=tensor([slots[w] for w in range(1,words+1) for i in range(words-w+1) for j in range(slots[w])])
        value=dict(slots=tuple(slots),layouts=layouts,unit_size=cid,bank=bank,widths=widths,root_slots=root_slots,root=root,unit_id=unit_id,slot_offset=slot_offset,slot_of=slot_of)
        CKYIndexCache._unit_geometry[key]=value;return value

    @staticmethod
    def _build_fast(words,batch,k,r,device,full_decode=False):
        u=CKYIndexCache._unit(words,k,r,device);tensors=[]
        rows=torch.arange(batch,device=device,dtype=torch.long)
        # Candidate IDs: span base scales with batch, then row*slots+slot.
        id_bank=torch.cat([(batch*u['unit_id']+u['slot_offset'])[None,:]+rows[:,None]*u['slot_of'][None,:],torch.full((batch,1),-1,device=device,dtype=torch.long)],1)
        widths=[None]
        for p in u['widths'][1:]:
            ud=p['unit_decode']  # (left span unit base, right span unit base, left slot, right slot, left slots, right slots)
            decode=torch.stack((batch*ud[...,0]+ud[...,2],batch*ud[...,1]+ud[...,3],ud[...,4],ud[...,5]),-1)
            tensors.extend((p['left'],p['right'],p['parent'],decode,p['ends']))
            widths.append(WidthIndex(p['width'],p['spans'],p['candidates'],p['budget'],p['keep'],p['meta'],p['left'],p['right'],p['parent'],decode,p['ends'],
                                     tuple(batch*b for b in p['unit_bases'])))
        tensors.append(u['root']);tensors.append(id_bank)
        if full_decode:
            for j,p in enumerate(widths):
                if p is None:continue
                left=id_bank.index_select(1,p.left.reshape(-1)).reshape(batch,p.spans,p.candidates)
                right=id_bank.index_select(1,p.right.reshape(-1)).reshape(batch,p.spans,p.candidates)
                tensors.extend((left,right));widths[j]=replace(p,left_ids=left,right_ids=right)
        layouts=tuple((batch*cid,i,j,kk) for cid,i,j,kk in u['layouts'])
        return ChartIndex(tuple(widths),u['slots'],layouts,batch*u['unit_size'],u['bank'],u['root_slots'],u['root'],id_bank,
                          sum(t.numel()*t.element_size() for t in tensors))

    @staticmethod
    def _build_reference(words,batch,k,r,device,full_decode=False):
        slots=[0,1];budgets=[0,0];counts=[0,0]
        for width in range(2,words+1):
            e=sum(slots[o]*slots[width-o] for o in range(1,width))
            budget=min(e,r) if r is not None else e
            slots.append(min(budget,k) if k is not None else budget);budgets.append(budget);counts.append(e)
        bank_bases={};candidate_bases={};layouts=[];bank=cid=0
        for width in range(1,words+1):
            bank_bases[width]=bank
            for i in range(words-width+1):
                candidate_bases[i,i+width]=cid;layouts.append((cid,i,i+width,slots[width]));cid+=batch*slots[width]
            bank+=(words-width+1)*slots[width]
        tensors=[]
        def tensor(data):
            t=torch.tensor(data,device=device,dtype=torch.long);tensors.append(t);return t
        widths=[None]
        for width in range(2,words+1):
            meta=tuple((o,l,rr) for o in range(1,width) for l in range(slots[o]) for rr in range(slots[width-o]))
            left=[];right=[];decode=[];spans=words-width+1
            for i in range(spans):
                left.append([bank_bases[o]+i*slots[o]+l for o,l,rr in meta])
                right.append([bank_bases[width-o]+(i+o)*slots[width-o]+rr for o,l,rr in meta])
                decode.append([(candidate_bases[i,i+o]+l,candidate_bases[i+o,i+width]+rr,slots[o],slots[width-o]) for o,l,rr in meta])
            widths.append(WidthIndex(width,spans,counts[width],budgets[width],slots[width],meta,
                tensor(left),tensor(right),tensor([i*words+i+width-1 for i in range(spans)]),tensor(decode),
                tensor([i+width for i in range(spans)]),tuple(candidate_bases[i,i+width] for i in range(spans))))
        # WidthIndex list uses width-1 indexing; width1 itself needs no pair plan.
        root_slots=max(slots);root=tensor([[bank_bases[w]+j if j<slots[w] else bank for j in range(root_slots)] for w in range(1,words+1)])
        ids=[]
        for row in range(batch):
            ids.append([candidate_bases[i,i+w]+row*slots[w]+j for w in range(1,words+1) for i in range(words-w+1) for j in range(slots[w])]+[-1])
        id_bank=tensor(ids)
        if full_decode:
            for j,p in enumerate(widths):
                if p is None:continue
                left=id_bank.index_select(1,p.left.reshape(-1)).reshape(batch,p.spans,p.candidates)
                right=id_bank.index_select(1,p.right.reshape(-1)).reshape(batch,p.spans,p.candidates)
                tensors.extend((left,right));widths[j]=replace(p,left_ids=left,right_ids=right)
        return ChartIndex(tuple(widths),tuple(slots),tuple(layouts),cid,bank,root_slots,root,id_bank,
                          sum(t.numel()*t.element_size() for t in tensors))
