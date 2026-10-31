"""V2.5: differentiable scalar CKY, then root-closure semantic DAG evaluation.

The score search deliberately mirrors Reference/V2 arithmetic and random draws.
Only discrete pointers/masks cross to CPU, once; no neural work occurs in DFS.
Candidate IDs are (span base + row * actual slots + slot), including invalid slots.
"""
from __future__ import annotations
from bisect import bisect_right
from dataclasses import dataclass
import time
import torch
from torch import nn
from torch.nn import functional as F
from models.forest_reference import masked_log_softmax

@dataclass
class ScoreCell:
    score: torch.Tensor
    mask: torch.Tensor
    base: int
    left: torch.Tensor | None = None
    right: torch.Tensor | None = None

class LazyForestMixin:
    collect_timing = False

    def forward(self, leaves, span_scores, span_coeff, word_mask, return_trees=False, *, return_spans=False, return_audit=False):
        if self.implementation != "lazy_semantic":
            return super().forward(leaves, span_scores, span_coeff, word_mask, return_trees,
                                   return_spans=return_spans, return_audit=return_audit)
        if self.training and any(isinstance(m, nn.modules.dropout._DropoutNd) and m.p > 0 or
                                 isinstance(m, nn.modules.batchnorm._BatchNorm) for m in self.composer.modules()):
            raise ValueError("Lazy evaluation requires a deterministic, stateless Composer in training")
        return_spans = return_spans or self.capture_spans
        search = self._search_scores(leaves,span_scores,word_mask,return_spans)
        return self._finish_lazy(leaves,span_coeff,search,return_trees,return_spans,return_audit)

    def _search_scores(self,leaves,span_scores,word_mask,return_spans=False):
        n,w,d = leaves.shape
        if not w:
            raise ValueError("Forest expects at least one padded word slot")
        timings = {}
        events = []
        def mark(name):
            if self.collect_timing and leaves.is_cuda:
                e = torch.cuda.Event(enable_timing=True); e.record(); events.append((name,e))
        mark("search_start")
        t0 = time.perf_counter()
        lengths = word_mask.sum(-1)
        chart, debug_spans, layouts, bands = {}, {}, [], []
        return_spans = return_spans or self.capture_spans
        total_candidates = leaves.new_zeros((), dtype=torch.float32)
        total_spans = leaves.new_zeros((), dtype=torch.float32)
        next_id, v2_rows = 0, 0
        for i in range(w):
            leaf_score = span_scores[:,i,i:i+1]
            mask = (lengths > i).unsqueeze(-1) & torch.isfinite(leaf_score)
            cell = ScoreCell(leaf_score.masked_fill(~mask,0.0), mask, next_id)
            chart[i,i+1] = cell
            layouts.append((next_id,i,i+1,1)); next_id += n
            blank = torch.full((n,),-1,device=leaves.device,dtype=torch.long)
            bands.append((blank,blank,mask.reshape(-1)))
            if return_spans:
                debug_spans[i,i+1] = dict(scores=cell.score,candidate_mask=mask,selected_flat=None,
                                        retained_flat=None,metadata=(),candidate_base=cell.base)
            total_candidates = total_candidates + mask.sum()
            total_spans = total_spans + mask.sum()
        for width in range(2, w + 1):
            starts = list(range(w - width + 1))
            # Same-width spans share child slot counts. Build scalar priorities
            # before invoking the neural composer, then batch selected pairs.
            scores, masks = [], []
            metadata = []
            for offset in range(1, width):
                left = [chart[i, i + offset] for i in starts]
                right = [chart[i + offset, i + width] for i in starts]
                ls, rs = torch.stack([x.score for x in left], 1), torch.stack([x.score for x in right], 1)
                lm, rm = torch.stack([x.mask for x in left], 1), torch.stack([x.mask for x in right], 1)
                kl, kr = ls.shape[-1], rs.shape[-1]
                scores.append((ls.unsqueeze(-1) + rs.unsqueeze(-2)).flatten(2, 3))
                masks.append((lm.unsqueeze(-1) & rm.unsqueeze(-2)).flatten(2, 3))
                metadata.extend((offset, l, r) for l in range(kl) for r in range(kr))
            score = torch.cat(scores, 2).float()
            valid = torch.cat(masks, 2)
            local = torch.stack([span_scores[:, i, i + width - 1] for i in starts], 1)
            score = score + local.unsqueeze(-1)
            # -inf denotes an unreachable potential. Both implementations share
            # this guard; finite-input search order remains identical to V0.
            valid = valid & torch.isfinite(score)
            score = score.masked_fill(~valid, -torch.inf)
            e = score.shape[-1]
            budget = min(e, self.max_compositions) if self.max_compositions is not None else e
            explore = min(budget - 1, max(1, round(budget * self.exploration))) if self.training and self.exploration > 0 and budget < e else 0
            priority = score.detach()
            chosen = priority.topk(budget - explore, dim=-1).indices
            if explore:
                noise = torch.rand_like(priority).masked_fill(~valid, -torch.inf)
                noise.scatter_(2, chosen, -torch.inf)
                chosen = torch.cat((chosen, noise.topk(explore, dim=-1).indices), -1)
            pair_mask = valid.gather(2, chosen)
            expanded_score = score.gather(2, chosen)
            observer=getattr(self,"precision_observer",None)
            if observer is not None:
                observer(width,expanded_score,pair_mask,min(budget,self.k) if self.k is not None else budget)
            v2_rows += n * len(starts) * budget
            keep = min(budget, self.k) if self.k is not None else budget
            retained_score, rank = expanded_score.topk(keep, dim=-1)
            if self.training and self.exploration > 0 and keep > 1 and budget > keep:
                # Exploration must survive INTO the beam. Merely composing a
                # low-score pair and then taking score top-k would never explore.
                eligible = pair_mask.clone()
                eligible.scatter_(2, rank[..., :-1], False)
                if self.delta is not None:
                    eligible &= expanded_score >= retained_score[..., :1] - self.delta
                random_priority = torch.rand_like(expanded_score).masked_fill(~eligible, -torch.inf)
                random_rank = random_priority.argmax(-1, keepdim=True)
                use_random = (torch.rand_like(retained_score[..., :1]) < self.exploration) & eligible.any(-1, keepdim=True)
                last = torch.where(use_random, random_rank, rank[..., -1:])
                rank = torch.cat((rank[..., :-1], last), -1)
                retained_score = expanded_score.gather(2, rank)
            retained_mask = pair_mask.gather(2, rank)
            if self.delta is not None:
                best = retained_score[..., :1]
                retained_mask = retained_mask & (retained_score >= best - self.delta)
            retained_score = retained_score.masked_fill(~retained_mask, 0.0)
            retained_flat = chosen.gather(2, rank)
            # Shape-only pointer lookup. Candidate IDs include span, row and slot.
            key = ("lazy", n, w, width, tuple(chart[0, off].score.shape[-1] for off in range(1, width)), score.device)
            table = self._lookup_cache.get(key)
            if table is None:
                table = torch.tensor([[(chart[i, i+off].base+l, chart[i+off, i+width].base+r,
                                       chart[i, i+off].score.shape[-1], chart[i+off, i+width].score.shape[-1])
                                      for off,l,r in metadata] for i in starts], device=score.device)
                self._lookup_cache[key] = table
            decoded = table.unsqueeze(0).expand(n,-1,-1,-1).gather(2, retained_flat.unsqueeze(-1).expand(-1,-1,-1,4))
            rows = torch.arange(n, device=score.device)[:,None,None]
            lp, rp = decoded[...,0]+rows*decoded[...,2], decoded[...,1]+rows*decoded[...,3]
            for s, i in enumerate(starts):
                cell = ScoreCell(retained_score[:,s], retained_mask[:,s], next_id, lp[:,s], rp[:,s])
                chart[i,i+width] = cell
                layouts.append((next_id, i, i+width, keep))
                next_id += n*keep
                if return_spans:
                    debug_spans[i,i+width] = dict(scores=cell.score, candidate_mask=cell.mask,
                        selected_flat=chosen[:,s], retained_flat=retained_flat[:,s], metadata=tuple(metadata),
                        left_id=cell.left, right_id=cell.right, candidate_base=cell.base)
            bands.append((lp.permute(1,0,2).reshape(-1), rp.permute(1,0,2).reshape(-1),
                          retained_mask.permute(1,0,2).reshape(-1)))
            total_candidates = total_candidates + retained_mask.sum()
            total_spans = total_spans + (lengths[:, None] >= torch.tensor([i + width for i in starts], device=lengths.device)).sum()

        root_slots = max(chart[0,L].score.shape[-1] for L in range(1,w+1))
        ss, ms, ids = [], [], []
        row = torch.arange(n,device=leaves.device)
        for L in range(1,w+1):
            cell = chart[0,L]; k = cell.score.shape[-1]; pad = root_slots-k
            ss.append(F.pad(cell.score,(0,pad)))
            ms.append(F.pad(cell.mask,(0,pad)))
            ids.append(F.pad(cell.base+row[:,None]*k+torch.arange(k,device=leaves.device),(0,pad),value=-1))
        index = lengths.clamp_min(1)-1
        root_s = torch.stack(ss,1)[row,index]
        root_m = torch.stack(ms,1)[row,index] & lengths.ne(0).unsqueeze(-1)
        root_ids = torch.stack(ids,1)[row,index]
        mark("search_end")
        timings["score_search_host_ms"] = (time.perf_counter()-t0)*1000
        return dict(lengths=lengths,layouts=layouts,bands=bands,next_id=next_id,
                    root_s=root_s,root_m=root_m,root_ids=root_ids,total_candidates=total_candidates,
                    total_spans=total_spans,v2_rows=v2_rows,debug_spans=debug_spans,
                    timings=timings,events=events,mark=mark)

    def _finish_lazy(self,leaves,span_coeff,search,return_trees,return_spans,return_audit):
        # Shared V2.5 closure/DAG/Composer code; the search implementation supplies scalars/IDs only.
        n,w,d=leaves.shape;c=span_coeff.shape[-1]
        lengths,layouts,bands,next_id=(search[k] for k in ("lengths","layouts","bands","next_id"))
        root_s,root_m,root_ids=(search[k] for k in ("root_s","root_m","root_ids"))
        root_slots=root_s.shape[-1]
        total_candidates,total_spans,v2_rows=(search[k] for k in ("total_candidates","total_spans","v2_rows"))
        debug_spans,timings,events,mark=(search[k] for k in ("debug_spans","timings","events","mark"))
        t0 = time.perf_counter()
        # One batched device->host read, containing only integer topology/masks.
        packed = torch.cat([torch.cat([b[q].reshape(-1) for b in bands]).long() for q in range(3)] +
                           [root_ids.reshape(-1),root_m.reshape(-1).long(),lengths])
        raw = packed.detach().cpu().tolist()
        timings["topology_d2h_blocking_ms"] = (time.perf_counter()-t0)*1000
        mark("topology_end")
        t0 = time.perf_counter()
        left, right, valid = raw[:next_id],raw[next_id:2*next_id],raw[2*next_id:3*next_id]
        root_cpu = raw[3*next_id:3*next_id+n*root_slots]
        masks = raw[3*next_id+n*root_slots:3*next_id+2*n*root_slots]
        lens = raw[-n:]
        bases = [x[0] for x in layouts]
        def decode(cid):
            base,i,j,k = layouts[bisect_right(bases,cid)-1]
            r,slot = divmod(cid-base,k)
            if cid < base or r >= n: raise RuntimeError("Invalid Candidate ID")
            return r,i,j,slot
        needed = set()
        stack = [cid for cid,m in zip(root_cpu,masks) if m]
        while stack:
            cid = stack.pop()
            if cid in needed: continue
            if cid < 0 or cid >= next_id or not valid[cid]:
                raise RuntimeError("Reachable DAG contains an invalid candidate")
            needed.add(cid)
            r,i,j,slot = decode(cid)
            if j-i > 1:
                l,rr = left[cid],right[cid]
                lr,li,lj,_ = decode(l); rrrow,ri,rj,_ = decode(rr)
                if not (lr == rrrow == r and li == i and lj == ri and rj == j and i < lj < j):
                    raise RuntimeError("Malformed or cyclic CKY backpointer")
                stack.extend((l,rr))
        layers = {}
        compact = {i*n+r:r*w+i for i in range(w) for r in range(n)}
        for cid in sorted(needed):
            r,i,j,slot = decode(cid)
            if j-i > 1: layers.setdefault(j-i,[]).append(cid)
        # Shape information comes from CPU layout; index tensors use one H2D upload.
        plan, offsets, position = [], [], n*w
        for width,nodes in sorted(layers.items()):
            for cid in nodes: compact[cid] = position; position += 1
            a = len(plan)
            plan.extend(compact[left[cid]] for cid in nodes)
            plan.extend(compact[right[cid]] for cid in nodes)
            plan.extend(decode(cid)[0]*w*w+decode(cid)[1]*w+decode(cid)[2]-1 for cid in nodes)
            offsets.append((width,len(nodes),a))
        root_offset = len(plan)
        emit = []
        for q,(cid,m) in enumerate(zip(root_cpu,masks)):
            # Legacy keeps the leaf tensor in slot zero even for an empty row.
            leaf_slot = lens[q//root_slots] <= 1 and q%root_slots == 0
            plan.append(compact[cid] if m or leaf_slot else 0)
            emit.append(bool(m or leaf_slot))
        occurrences = sum(max(0,lens[q//root_slots]-1) for q,m in enumerate(masks) if m)
        internal = sum(len(nodes) for nodes in layers.values())
        timings["dag_cpu_ms"] = (time.perf_counter()-t0)*1000
        mark("dag_end")
        t0 = time.perf_counter()
        indices = torch.tensor(plan,device=leaves.device,dtype=torch.long)
        emission = torch.tensor(emit,device=leaves.device).reshape(n,root_slots,1)
        timings["index_h2d_host_ms"] = (time.perf_counter()-t0)*1000
        mark("index_end")
        t0 = time.perf_counter()
        hs = [leaves.reshape(n*w,d)]
        us = [span_coeff.diagonal(dim1=1,dim2=2).transpose(1,2).reshape(n*w,c)]
        flat_u = span_coeff.reshape(n*w*w,c)
        for width,count,a in offsets:
            hbank,ubank = torch.cat(hs,0),torch.cat(us,0)
            li,ri,ui = indices[a:a+count],indices[a+count:a+2*count],indices[a+2*count:a+3*count]
            if getattr(self,"precision_diagnostics",False): self.composer._forest_width=width
            hs.append(self._compose(hbank.index_select(0,li),hbank.index_select(0,ri)))
            us.append((ubank.index_select(0,li)+ubank.index_select(0,ri))+flat_u.index_select(0,ui))
        roots = indices[root_offset:]
        root_h = torch.cat(hs,0).index_select(0,roots).reshape(n,root_slots,d)*emission
        root_u = torch.cat(us,0).index_select(0,roots).reshape(n,root_slots,c)*emission
        if not internal and w > 1:
            # Eager V2 creates zero (not None) Composer grads on empty closures.
            # Preserve AdamW reachability without evaluating any unused node.
            root_h = root_h + sum(p.reshape(-1)[0]*0 for p in self.composer.parameters())
        mark("semantic_end")
        timings["semantic_host_ms"] = (time.perf_counter()-t0)*1000
        if self.collect_timing and leaves.is_cuda:
            events[-1][1].synchronize()
            es = dict(events)
            timings.update(score_search_cuda_ms=es["search_start"].elapsed_time(es["search_end"]),
                           index_h2d_cuda_ms=es["dag_end"].elapsed_time(es["index_end"]),
                           semantic_cuda_ms=es["index_end"].elapsed_time(es["semantic_end"]))
        self.last_workload = dict(v2_composer_candidate_rows=v2_rows,
            v25_required_node_occurrences=occurrences,v25_unique_dag_nodes=internal,
            v25_unique_leaf_nodes=len(needed)-internal,v25_actual_composer_rows=internal,
            v25_composer_calls=len(layers),v25_unused_candidates_skipped=v2_rows-internal,
            sharing_ratio=1-internal/occurrences if occurrences else 0,
            per_width_active_rows=[dict(width=x,rows=len(v)) for x,v in sorted(layers.items())],
            topology_bytes=packed.numel()*8,index_bytes=len(plan)*8+len(emit),**timings)
        result = dict(semantic=root_h,scores=root_s,vector_coeff=root_u,candidate_mask=root_m,
            log_prior=masked_log_softmax(root_s,root_m),diagnostics=dict(
                mean_span_candidates=total_candidates/total_spans.clamp_min(1),
                span_candidates_sum=total_candidates,span_count=total_spans))
        if return_spans: result["spans"] = debug_spans
        if return_audit or self.collect_audit: result["materialization_audit"] = self.last_workload.copy()
        if return_trees:
            trees = {}
            for cid in sorted(needed):
                r,i,j,slot = decode(cid)
                trees[cid] = i if j-i == 1 else (trees[left[cid]],trees[right[cid]])
            result["trees"] = [[trees[root_cpu[r*root_slots+k]] for k in range(root_slots)
                                if masks[r*root_slots+k]] for r in range(n)]
        return result
