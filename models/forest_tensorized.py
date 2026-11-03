"""V3-A/B compact-ragged wavefront scoring; unchanged native selection and shared DAG."""
from __future__ import annotations
from contextlib import contextmanager,nullcontext
import time
import torch
from torch import nn
from torch.nn import functional as F
from models.cky_index_cache import CKYIndexCache
from models.forest_lazy import ScoreCell

SCORE_RECONSTRUCT='lazy_tensorized_score_reconstruct'
TENSORIZED_IMPLEMENTATIONS=('lazy_tensorized_a','lazy_tensorized',SCORE_RECONSTRUCT)

class TensorizedForestMixin:
    def _apply(self,fn,recurse=True):
        if hasattr(self,'_geometry_cache'):self._geometry_cache.clear()
        return super()._apply(fn,recurse)

    def forward(self,leaves,span_scores,span_coeff,word_mask,return_trees=False,*,return_spans=False,return_audit=False):
        if self.implementation not in TENSORIZED_IMPLEMENTATIONS:
            return super().forward(leaves,span_scores,span_coeff,word_mask,return_trees,return_spans=return_spans,return_audit=return_audit)
        if self.training and any(isinstance(m,nn.modules.dropout._DropoutNd) and m.p>0 or
                                 isinstance(m,nn.modules.batchnorm._BatchNorm) for m in self.composer.modules()):
            raise ValueError('Lazy evaluation requires a deterministic, stateless Composer in training')
        return_spans=return_spans or self.capture_spans
        search=self._search_scores(leaves,span_scores,word_mask,return_spans)
        result=self._finish_lazy(leaves,span_coeff,search,return_trees,return_spans,return_audit)
        return self._record_tensorized_audit(search,result,return_audit)

    def _record_tensorized_audit(self,search,result,return_audit):
        if 'v3' not in search:
            self.last_workload['search_compatibility_fallback']='exact_unbounded'
            if return_audit or self.collect_audit:result['materialization_audit']=self.last_workload.copy()
            return result
        extra=search['v3'];self.last_workload.update(extra['cache'])
        if self.collect_timing:
            rows=[]
            for width,row in extra['widths'].items():
                record=dict(row);record['valid_candidates']=int(record['valid_candidates'].cpu())
                record.update({key:a.elapsed_time(b) for ww,key,a,b in extra['events'] if ww==width})
                rows.append(record)
            self.last_workload['width_profile']=rows
        if return_audit or self.collect_audit:result['materialization_audit']=self.last_workload.copy()
        return result

    def _search_scores(self,leaves,span_scores,word_mask,return_spans=False):
        if self.implementation not in TENSORIZED_IMPLEMENTATIONS:
            return super()._search_scores(leaves,span_scores,word_mask,return_spans)
        if self.k is None and self.max_compositions is None:
            # Exact Catalan enumeration is outside the bounded beam fast path.
            # Keep legacy score accumulation order in its cancellation-sensitive gradients.
            return super()._search_scores(leaves,span_scores,word_mask,return_spans)
        n,w,d=leaves.shape
        if not w:raise ValueError('Forest expects at least one padded word slot')
        if not hasattr(self,'_geometry_cache'):self._geometry_cache=CKYIndexCache()
        timings={};events=[];stage_events=[];width_rows={}
        def mark(name):
            if self.collect_timing and leaves.is_cuda:
                event=torch.cuda.Event(enable_timing=True);event.record();events.append((name,event))
        @contextmanager
        def region(width,name):
            active=self.collect_timing and leaves.is_cuda
            if active:
                a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);a.record()
            ctx=torch.profiler.record_function(f'V3/width_{width}/{name}') if getattr(self,'v3_profile',False) else nullcontext()
            with ctx:yield
            if active:b.record();stage_events.append((width,name,a,b))
        mark('search_start');start=time.perf_counter()
        fast=self.implementation!='lazy_tensorized_a'
        backend=getattr(self,'search_backend','eager')
        if backend!='eager':
            if not fast:raise ValueError('search_backend requires the full-decode (lazy_tensorized) layout')
            if getattr(self,'precision_observer',None) is not None:raise ValueError('precision_observer is an eager-only diagnostic')
            from models.cky_compile import get_width_step,mark_rows_dynamic,begin_step
            step_fn=get_width_step(backend);begin_step(backend)
        plan,hit,build_ms=self._geometry_cache.get(w,n,self.k,self.max_compositions,leaves.device,full_decode=fast)
        lengths=word_mask.sum(-1);chart={};debug_spans={};bands=[]
        leaf_score=span_scores.diagonal(dim1=1,dim2=2)
        leaf_mask=(lengths[:,None]>torch.arange(w,device=leaves.device))&torch.isfinite(leaf_score)
        leaf_score=leaf_score.masked_fill(~leaf_mask,0.)
        scores_by_width=[leaf_score.unsqueeze(-1)];masks_by_width=[leaf_mask.unsqueeze(-1)]
        total_candidates=leaf_mask.sum().float();total_spans=leaf_mask.sum().float();v2_rows=0
        if not fast or return_spans:
            for i in range(w):
                if not fast:chart[i,i+1]=ScoreCell(leaf_score[:,i:i+1],leaf_mask[:,i:i+1],i*n)
                if return_spans:debug_spans[i,i+1]=dict(scores=leaf_score[:,i:i+1],candidate_mask=leaf_mask[:,i:i+1],selected_flat=None,retained_flat=None,metadata=(),candidate_base=i*n)
        blank=torch.full((n*w,),-1,device=leaves.device,dtype=torch.long)
        bands.append((blank,blank,leaf_mask.transpose(0,1).reshape(-1)))
        span_flat=span_scores.flatten(1)
        for width in range(2,w+1):
            p=plan.widths[width-1]
            if backend!='eager':
                # Same arithmetic as the inline loop below, as one pure function (plain or compiled).
                with region(width,'elapsed_ms'):
                    bank=torch.cat([x.reshape(n,-1) for x in scores_by_width],1)
                    mask_bank=torch.cat([x.reshape(n,-1) for x in masks_by_width],1)
                    budget=p.budget;keep=p.keep;e=p.candidates
                    explore=min(budget-1,max(1,round(budget*self.exploration))) if self.training and self.exploration>0 and budget<e else 0
                    swap=bool(self.training and self.exploration>0 and keep>1 and budget>keep)
                    if backend!='function':mark_rows_dynamic(bank,mask_bank,span_flat,p.left_ids,p.right_ids)  # per-row candidate-id planes are row-indexed too
                    retained_score,retained_mask,lp,rp,chosen,retained_flat,valid=step_fn(bank,mask_bank,span_flat,p.left,p.right,p.parent,p.left_ids,p.right_ids,p.spans,e,budget,keep,explore,swap,self.delta,self.exploration)
                    scores_by_width.append(retained_score);masks_by_width.append(retained_mask)
                    if return_spans:
                        for i in range(p.spans):
                            debug_spans[i,i+width]=dict(scores=retained_score[:,i],candidate_mask=retained_mask[:,i],selected_flat=chosen[:,i],retained_flat=retained_flat[:,i],metadata=p.metadata,left_id=lp[:,i],right_id=rp[:,i],candidate_base=p.bases[i])
                    bands.append((lp.permute(1,0,2).reshape(-1),rp.permute(1,0,2).reshape(-1),retained_mask.permute(1,0,2).reshape(-1)))
                    total_candidates=total_candidates+retained_mask.sum()
                    total_spans=total_spans+(lengths[:,None]>=p.ends).sum()
                    v2_rows+=n*p.spans*budget
                if self.collect_timing:width_rows[width]=dict(width=width,candidates=n*p.spans*e,valid_candidates=valid.sum(),spans=p.spans,slots=keep,legacy_capacity=e)
                continue
            with region(width,'elapsed_ms'):
                with region(width,'gather_ms'):
                    # Width2 retains the original BF16 leaf addition. All later
                    # pairs have >=one FP32 child, matching legacy promotion.
                    bank=torch.cat([x.reshape(n,-1) for x in scores_by_width],1)
                    mask_bank=torch.cat([x.reshape(n,-1) for x in masks_by_width],1)
                    ls=bank.index_select(1,p.left.reshape(-1)).reshape(n,p.spans,p.candidates)
                    rs=bank.index_select(1,p.right.reshape(-1)).reshape(n,p.spans,p.candidates)
                    lm=mask_bank.index_select(1,p.left.reshape(-1)).reshape(n,p.spans,p.candidates)
                    rm=mask_bank.index_select(1,p.right.reshape(-1)).reshape(n,p.spans,p.candidates)
                    local=span_scores.flatten(1).index_select(1,p.parent)
                with region(width,'pair_ms'):
                    score=(ls+rs).float()+local.unsqueeze(-1)
                    valid=(lm&rm)&torch.isfinite(score);score=score.masked_fill(~valid,-torch.inf)
                with region(width,'selection_ms'):
                    budget=p.budget;keep=p.keep;e=p.candidates
                    explore=min(budget-1,max(1,round(budget*self.exploration))) if self.training and self.exploration>0 and budget<e else 0
                    priority=score.detach();chosen=priority.topk(budget-explore,dim=-1).indices
                    if explore:
                        noise=torch.rand_like(priority).masked_fill(~valid,-torch.inf);noise.scatter_(2,chosen,-torch.inf)
                        chosen=torch.cat((chosen,noise.topk(explore,dim=-1).indices),-1)
                    pair_mask=valid.gather(2,chosen);expanded_score=score.gather(2,chosen)
                    observer=getattr(self,'precision_observer',None)
                    if observer is not None:observer(width,expanded_score,pair_mask,keep)
                    retained_score,rank=expanded_score.topk(keep,dim=-1)
                    if self.training and self.exploration>0 and keep>1 and budget>keep:
                        eligible=pair_mask.clone();eligible.scatter_(2,rank[...,:-1],False)
                        if self.delta is not None:eligible&=expanded_score>=retained_score[...,:1]-self.delta
                        random_priority=torch.rand_like(expanded_score).masked_fill(~eligible,-torch.inf)
                        random_rank=random_priority.argmax(-1,keepdim=True)
                        use_random=(torch.rand_like(retained_score[...,:1])<self.exploration)&eligible.any(-1,keepdim=True)
                        last=torch.where(use_random,random_rank,rank[...,-1:]);rank=torch.cat((rank[...,:-1],last),-1)
                        retained_score=expanded_score.gather(2,rank)
                    retained_mask=pair_mask.gather(2,rank)
                    if self.delta is not None:retained_mask=retained_mask&(retained_score>=retained_score[...,:1]-self.delta)
                    retained_score=retained_score.masked_fill(~retained_mask,0.)
                    retained_flat=chosen.gather(2,rank)
                with region(width,'decode_ms'):
                    if fast:
                        lp=p.left_ids.gather(2,retained_flat);rp=p.right_ids.gather(2,retained_flat)
                    else:
                        decoded=p.decode.unsqueeze(0).expand(n,-1,-1,-1).gather(2,retained_flat.unsqueeze(-1).expand(-1,-1,-1,4))
                        rows=torch.arange(n,device=leaves.device)[:,None,None]
                        lp,rp=decoded[...,0]+rows*decoded[...,2],decoded[...,1]+rows*decoded[...,3]
                with region(width,'assembly_ms'):
                    scores_by_width.append(retained_score);masks_by_width.append(retained_mask)
                    if not fast or return_spans:
                        for i in range(p.spans):
                            if not fast:chart[i,i+width]=ScoreCell(retained_score[:,i],retained_mask[:,i],p.bases[i],lp[:,i],rp[:,i])
                            if return_spans:debug_spans[i,i+width]=dict(scores=retained_score[:,i],candidate_mask=retained_mask[:,i],selected_flat=chosen[:,i],retained_flat=retained_flat[:,i],metadata=p.metadata,left_id=lp[:,i],right_id=rp[:,i],candidate_base=p.bases[i])
                    bands.append((lp.permute(1,0,2).reshape(-1),rp.permute(1,0,2).reshape(-1),retained_mask.permute(1,0,2).reshape(-1)))
                    total_candidates=total_candidates+retained_mask.sum()
                    total_spans=total_spans+(lengths[:,None]>=p.ends).sum()
                    v2_rows+=n*p.spans*budget
                if self.collect_timing:width_rows[width]=dict(width=width,candidates=n*p.spans*e,valid_candidates=valid.sum(),spans=p.spans,slots=keep,legacy_capacity=e)
        if fast:
            index=lengths.clamp_min(1)-1
            gather=plan.root_gather.index_select(0,index)
            bank=torch.cat([x.reshape(n,-1) for x in scores_by_width]+[leaf_score.new_zeros(n,1)],1)
            mask_bank=torch.cat([x.reshape(n,-1) for x in masks_by_width]+[leaf_mask.new_zeros(n,1)],1)
            root_s=bank.gather(1,gather);root_m=mask_bank.gather(1,gather)&lengths.ne(0).unsqueeze(-1)
            root_ids=plan.id_bank.gather(1,gather)
        else:
            root_slots=plan.root_slots;ss=[];ms=[];ids=[];row=torch.arange(n,device=leaves.device)
            for length in range(1,w+1):
                cell=chart[0,length];k=cell.score.shape[-1];pad=root_slots-k
                ss.append(F.pad(cell.score,(0,pad)));ms.append(F.pad(cell.mask,(0,pad)))
                ids.append(F.pad(cell.base+row[:,None]*k+torch.arange(k,device=leaves.device),(0,pad),value=-1))
            index=lengths.clamp_min(1)-1
            root_s=torch.stack(ss,1)[row,index];root_m=torch.stack(ms,1)[row,index]&lengths.ne(0).unsqueeze(-1)
            root_ids=torch.stack(ids,1)[row,index]
        mark('search_end');timings['score_search_host_ms']=(time.perf_counter()-start)*1000
        cache=dict(index_cache_hit=hit,index_cache_build_host_ms=build_ms,index_cache_bytes=plan.device_bytes,
                   index_cache_hits=self._geometry_cache.hits,index_cache_misses=self._geometry_cache.misses,
                   index_cache_hit_rate=self._geometry_cache.hits/max(1,self._geometry_cache.hits+self._geometry_cache.misses))
        return dict(lengths=lengths,layouts=plan.layouts,bands=bands,next_id=plan.size,
                    root_s=root_s,root_m=root_m,root_ids=root_ids,total_candidates=total_candidates,total_spans=total_spans,
                    v2_rows=v2_rows,debug_spans=debug_spans,timings=timings,events=events,mark=mark,
                    v3=dict(cache=cache,widths=width_rows,events=stage_events))
