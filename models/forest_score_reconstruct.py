"""V3-C: no-grad V3 score search + differentiable root-score reconstruction along the lazy DAG.

Hard Top-R/Top-K selection is discrete; only the *selected* root derivations carry
score gradient in V3 (gather backward zeroes every other candidate). V3-C therefore
runs the identical width-batched search without autograd, then recomputes the
selected root scores from the original differentiable span potentials along the
same DAG plan that the lazy tail already builds for the Composer:

    l_P = (l_L + l_R).float() + s_{i:j}

This is the exact operation sequence of the V3 pairing step, so forward root
scores are bitwise identical (BF16 leaf+leaf addition at width 2 included).
Candidate IDs, backpointers, masks, RNG draws, Composer/Reader/Decoder code are
untouched. Modes:
  dag    (C1) shared DAG nodes, each computed once, gradients accumulate through sharing
  tree   (C2) per-root-tree recomputation without sharing; correctness reference
  linear (C3) root score as one flat sum of the tree's span potentials; changes the
              floating-point addition order and is NOT forward-bitwise. Experimental,
              reported separately; never selected by configs.
"""
from __future__ import annotations
import torch
from torch import nn
from models.forest_tensorized import SCORE_RECONSTRUCT

RECONSTRUCTION_MODES=('dag','tree','linear')


class ScoreReconstructForestMixin:
    reconstruction_mode='dag'

    def forward(self,leaves,span_scores,span_coeff,word_mask,return_trees=False,*,return_spans=False,return_audit=False):
        if self.implementation!=SCORE_RECONSTRUCT:
            return super().forward(leaves,span_scores,span_coeff,word_mask,return_trees,return_spans=return_spans,return_audit=return_audit)
        if self.k is None and self.max_compositions is None:
            # Exact Catalan enumeration keeps the legacy differentiable search (V3 behaviour).
            return super().forward(leaves,span_scores,span_coeff,word_mask,return_trees,return_spans=return_spans,return_audit=return_audit)
        if self.training and any(isinstance(m,nn.modules.dropout._DropoutNd) and m.p>0 or
                                 isinstance(m,nn.modules.batchnorm._BatchNorm) for m in self.composer.modules()):
            raise ValueError('Lazy evaluation requires a deterministic, stateless Composer in training')
        if self.reconstruction_mode not in RECONSTRUCTION_MODES:raise ValueError(self.reconstruction_mode)
        return_spans=return_spans or self.capture_spans
        with torch.no_grad():
            search=self._search_scores(leaves,span_scores,word_mask,return_spans)
        search['reconstruct']=self._score_reconstructor(span_scores,word_mask,search)
        result=self._finish_lazy(leaves,span_coeff,search,return_trees,return_spans,return_audit)
        self.last_workload['score_reconstruction_mode']=self.reconstruction_mode
        self.last_workload['score_reconstruction_nodes']=self._reconstruction_nodes
        return self._record_tensorized_audit(search,result,return_audit)

    def _score_reconstructor(self,span_scores,word_mask,search):
        n,w=word_mask.shape;lengths=search['lengths'];mode=self.reconstruction_mode
        leaf=span_scores.diagonal(dim1=1,dim2=2)
        leaf_mask=(lengths[:,None]>torch.arange(w,device=leaf.device))&torch.isfinite(leaf)
        leaf=leaf.masked_fill(~leaf_mask,0.)  # Same leaf bank values as the V3 search.
        # Exact upcast. V3 adds BF16 `local` to an FP32 sum by type promotion, i.e. in
        # FP32; gathering the FP32 view keeps that arithmetic and makes the backward
        # accumulate each span's candidate gradients in FP32 before one BF16 cast,
        # as V3 does per width.
        flat=span_scores.float().reshape(n*w*w)
        def reconstruct(ctx):
            if mode=='dag':return self._dag_reconstruct(ctx,leaf,flat,n,w)
            if mode=='tree':return self._tree_reconstruct(ctx,leaf,flat,n,w)
            return self._linear_reconstruct(ctx,leaf,flat,n,w)
        return reconstruct

    def _recurrence(self,indices,offsets,root_offset,root_m,leaf,flat,n,w):
        ss=[leaf.reshape(n*w)]
        for width,count,a in offsets:
            bank=torch.cat(ss,0)
            li,ri,ui=indices[a:a+count],indices[a+count:a+2*count],indices[a+2*count:a+3*count]
            # Width 2: BF16 leaf + BF16 leaf rounded to BF16, then .float(). Later widths:
            # the cat-promoted FP32 bank, exactly as the V3 width loop's `bank`.
            ss.append((bank.index_select(0,li)+bank.index_select(0,ri)).float()+flat.index_select(0,ui))
        bank=torch.cat(ss,0) if len(ss)>1 else ss[0]
        root_s=bank.index_select(0,indices[root_offset:]).reshape(n,root_m.shape[-1])
        if w>1:root_s=root_s.float()  # V3 root bank is FP32 whenever a width>=2 layer exists.
        self._reconstruction_nodes=sum(c for _,c,_ in offsets)
        return root_s.masked_fill(~root_m,0.)

    def _dag_reconstruct(self,ctx,leaf,flat,n,w):
        return self._recurrence(ctx['indices'],ctx['offsets'],ctx['root_offset'],ctx['root_m'],leaf,flat,n,w)

    def _tree_reconstruct(self,ctx,leaf,flat,n,w):
        """C2 reference: expand every root tree into unshared occurrences, then the same recurrence."""
        left,right,root_cpu,masks,lens,decode,root_slots=(ctx[k] for k in ('left','right','root_cpu','masks','lens','decode','root_slots'))
        entries=[]  # One entry per internal-node OCCURRENCE: (width, left ref, right ref, parent span flat index).
        def expand(cid):
            r,i,j,_=decode(cid)
            if j-i==1:return ('leaf',r*w+i)
            l=expand(left[cid]);rr=expand(right[cid])
            entries.append((j-i,l,rr,r*w*w+i*w+j-1));return ('node',len(entries)-1)
        roots=[]
        for q,(cid,m) in enumerate(zip(root_cpu,masks)):
            leaf_slot=lens[q//root_slots]<=1 and q%root_slots==0
            roots.append(expand(cid) if m or leaf_slot else ('leaf',0))
        by_width={}
        for e,(width,l,rr,u) in enumerate(entries):by_width.setdefault(width,[]).append(e)
        compact={};position=n*w;plan=[];offsets=[]
        for width,es in sorted(by_width.items()):
            for e in es:compact[e]=position;position+=1
        def pos(ref):return ref[1] if ref[0]=='leaf' else compact[ref[1]]
        for width,es in sorted(by_width.items()):
            a=len(plan)
            plan.extend(pos(entries[e][1]) for e in es);plan.extend(pos(entries[e][2]) for e in es);plan.extend(entries[e][3] for e in es)
            offsets.append((width,len(es),a))
        root_offset=len(plan);plan.extend(pos(r) for r in roots)
        indices=torch.tensor(plan,device=leaf.device,dtype=torch.long)
        return self._recurrence(indices,offsets,root_offset,ctx['root_m'],leaf,flat,n,w)

    def _linear_reconstruct(self,ctx,leaf,flat,n,w):
        """C3 experimental: root score = flat FP32 sum over the tree's span potentials (order changed)."""
        left,right,root_cpu,masks,lens,decode,root_slots=(ctx[k] for k in ('left','right','root_cpu','masks','lens','decode','root_slots'))
        memo={}
        def spans(cid):
            if cid in memo:return memo[cid]
            r,i,j,_=decode(cid)
            out=[r*w*w+i*w+j-1] if j-i>1 else []
            if j-i>1:out=spans(left[cid])+spans(right[cid])+out
            memo[cid]=out;return out
        rows=[];leaf_rows=[];width=0
        for q,(cid,m) in enumerate(zip(root_cpu,masks)):
            if not m:rows.append([]);leaf_rows.append([]);continue
            r,i,j,_=decode(cid);rows.append(spans(cid));leaf_rows.append([r*w+t for t in range(i,j)]);width=max(width,len(rows[-1]),len(leaf_rows[-1]))
        width=max(width,1)
        def pad(lists):
            idx=torch.zeros(len(lists),width,dtype=torch.long);mask=torch.zeros(len(lists),width,dtype=torch.bool)
            for q,l in enumerate(lists):
                if l:idx[q,:len(l)]=torch.tensor(l);mask[q,:len(l)]=True
            return idx.to(leaf.device).reshape(-1),mask.to(leaf.device)
        si,sm=pad(rows);li,lm=pad(leaf_rows)
        internal=(flat.index_select(0,si).reshape(len(rows),width)*sm).sum(-1)
        leaves=(leaf.reshape(n*w).float().index_select(0,li).reshape(len(rows),width)*lm).sum(-1)
        self._reconstruction_nodes=int(sm.sum())
        root_s=(internal+leaves).reshape(n,root_slots)
        return root_s.masked_fill(~ctx['root_m'],0.)
