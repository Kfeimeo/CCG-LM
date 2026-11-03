"""Aggregate V3.5 audit/benchmark/training outputs into tables (markdown fragments + CSV)."""
from __future__ import annotations
import argparse,json,statistics
from pathlib import Path
import pandas as pd


def md(df,floatfmt='.2f'):
    cols=list(df.columns);lines=['| '+' | '.join(str(c) for c in cols)+' |','|'+'|'.join('---' for _ in cols)+'|']
    for _,r in df.iterrows():
        cells=[]
        for c in cols:
            v=r[c]
            if isinstance(v,float):cells.append('' if pd.isna(v) else format(v,floatfmt) if abs(v)<1e6 else format(v,'.3g'))
            else:cells.append(str(v))
        lines.append('| '+' | '.join(cells)+' |')
    return '\n'.join(lines)


def audit_tables(root,out):
    frag=[];audit=root/'audit'
    sweep=pd.read_csv(audit/'sweep.csv')
    cols=['micro_batch_size','status','update_wall_median_ms','update_wall_min_ms','micro_batches_per_update_median','micro_batch_wall_median_ms','ce_tokens_per_second','peak_allocated_gib','mean_padded_words','mean_word_padding_ratio']
    frag.append('### Micro-batch sweep (real data, 4096 CE tokens per update)\n\n'+md(sweep[[c for c in cols if c in sweep.columns]]))
    stage=['forward_cuda_ms_per_update','module/syntax_cuda_ms_per_update','module/forest_cuda_ms_per_update','module/decoder_cuda_ms_per_update','module/reader_cuda_ms_per_update','backward_cuda_ms_per_update','clip_cuda_ms_per_update','optimizer_cuda_ms_per_update','sampling_collation_ms_per_update','h2d_host_ms_per_update']
    have=[c for c in stage if c in sweep.columns]
    s2=sweep[['micro_batch_size','update_wall_median_ms']+have].copy()
    for c in have:s2[c.replace('_cuda_ms_per_update','').replace('_host_ms_per_update','').replace('_ms_per_update','')+'_%']=100*s2[c]/s2['update_wall_median_ms']
    frag.append('### Stage envelopes per update (CUDA event envelopes; module hooks nest inside forward)\n\n'+md(s2[['micro_batch_size','update_wall_median_ms']+have]))
    frag.append('### Stage share of update wall (%)\n\n'+md(s2[['micro_batch_size']+[c for c in s2.columns if c.endswith('_%')]],'.1f'))
    # Per-micro-batch cache deltas (counters are cumulative across the sweep).
    det=json.loads((audit/'sweep_updates.json').read_text(encoding='utf-8'))['details'];rows=[];prev=(0,0,0.)
    for mbs in sorted(det,key=int):
        d=det[mbs]
        if not d['updates']:rows.append(dict(micro_batch_size=int(mbs),status=d['status']));continue
        c=d['updates'][-1]['cache'];misses=c['misses']-prev[1];hits=c['hits']-prev[0];build=c['build_host_ms_total']-prev[2];prev=(c['hits'],c['misses'],c['build_host_ms_total'])
        n_updates=len(d['updates']);walls=[u['wall_ms'] for u in d['updates']]
        rows.append(dict(micro_batch_size=int(mbs),status=d['status'],timed_updates=n_updates,timed_hits=hits,timed_misses=misses,timed_hit_rate=hits/max(1,hits+misses),build_ms_per_update=build/n_updates,build_share_of_update=build/max(1e-9,sum(walls))))
    frag.append('### Index-cache behaviour during the timed sweep updates (LRU 8, old builder)\n\n'+md(pd.DataFrame(rows),'.3f'))
    cache=pd.read_csv(audit/'cache_simulation.csv');piv=cache.pivot(index='micro_batch_size',columns='capacity',values='hit_rate').reset_index()
    piv.columns=['micro_batch_size']+[f'lru{c}_hit_rate' for c in piv.columns[1:]]
    stream=pd.read_csv(audit/'length_distribution.csv')
    frag.append('### Real key stream (100 updates): unique keys, LRU hit rates by capacity\n\n'+md(stream[['micro_batch_size','micro_batches_per_update','unique_cache_keys']].merge(piv,on='micro_batch_size'),'.3f'))
    frag.append('### Padding / utilization of the real stream\n\n'+md(stream[['micro_batch_size','padded_words_mean','history_slots_mean','target_length_padded_mean','word_padding_ratio_mean','span_utilization_mean','target_padding_ratio_mean','duplicate_history_fraction']],'.3f'))
    dec=pd.read_csv(audit/'decomposition.csv')
    base=['micro_batch_size','batch_index','ce_tokens','padded_words','plain_wall_median_ms','plain_cuda_envelope_median_ms','ce_tokens_per_second_microbatch','profiled_wall_ms','profiled_gpu_busy_ms','profiled_gpu_idle_fraction','cuda_launches','graph_score_prior_autograd_nodes','graph_loss_autograd_nodes']
    frag.append('### Micro-batch decomposition: wall vs GPU busy (profiled run) and launches\n\n'+md(dec[[c for c in base if c in dec.columns]]))
    hooked=[c for c in dec.columns if (c.startswith('module/') or c.startswith('forest/') or c in ('h2d_cuda_ms','forward_cuda_ms','backward_cuda_ms','clip_cuda_ms','optimizer_cuda_ms')) and c.endswith('_cuda_ms')]
    frag.append('### Module/stage CUDA envelopes per micro-batch (hooked run, no internal sync)\n\n'+md(dec[['micro_batch_size','batch_index']+hooked]))
    forest=[c for c in dec.columns if c.startswith('forest_') and 'ms' in c]+['d2h_gpu_wait_ms_when_synced','d2h_copy_tolist_ms_when_synced','forest_v25_actual_composer_rows','forest_v25_composer_calls']
    frag.append('### Forest internals (collect_timing run) and D2H wait/copy split\n\n'+md(dec[['micro_batch_size','batch_index']+[c for c in forest if c in dec.columns]]))
    ka=[c for c in dec.columns if c.startswith('kernel_active_') and not c.startswith('kernel_active_V3/')]
    frag.append('### Kernel-active time attributed to ranges (profiled, ms)\n\n'+md(dec[['micro_batch_size','batch_index']+ka],'.2f'))
    v3=[c for c in dec.columns if c.startswith('kernel_active_V3/')]
    if v3:
        agg=dec[['micro_batch_size','batch_index']].copy()
        for stage_name in ('gather_ms','pair_ms','selection_ms','decode_ms','assembly_ms'):
            agg['search_'+stage_name.replace('_ms','')+'_kernel_ms']=dec[[c for c in v3 if c.endswith(f'/{stage_name}_ms')]].sum(axis=1)
            agg['search_'+stage_name.replace('_ms','')+'_launches']=dec[[c for c in dec.columns if c.startswith('launches_V3/') and c.endswith(f'/{stage_name}')]].sum(axis=1)
        frag.append('### Score-search kernel-active time and launches by sub-stage (sum over widths)\n\n'+md(agg,'.2f'))
    staged=[c for c in dec.columns if c.startswith('staged_') and c.endswith('_cuda_ms')]
    frag.append('### Staged (detached-boundary, synchronized) phase envelopes per micro-batch\n\n'+md(dec[['micro_batch_size','batch_index']+staged]))
    sk=[c for c in dec.columns if c.startswith('staged_kernel_active_')]
    frag.append('### Staged phases: kernel-active (profiled) ms\n\n'+md(dec[['micro_batch_size','batch_index']+sk],'.2f'))
    py=[c for c in dec.columns if c.startswith('python_')]
    frag.append('### Python exclusive time per micro-batch step (cProfile, ms)\n\n'+md(dec[['micro_batch_size','batch_index']+py]))
    (out/'audit_tables.md').write_text('\n\n'.join(frag),encoding='utf-8');print('audit tables written')


def benchmark_tables(root,out):
    frag=[];b=root/'benchmark'
    if (b/'benchmark.csv').exists():
        df=pd.read_csv(b/'benchmark.csv')
        cols=['case','arm','ce_tokens','step_median_ms','step_min_ms','ce_tokens_per_second','peak_allocated_gib','score_forward_median_ms','score_backward_median_ms','forest_forward_median_ms','reconstruction_cuda_ms','score_autograd_nodes','full_cuda_launches','gpu_idle_fraction','d2h_blocking_ms','dag_cpu_ms']
        frag.append('### Paired benchmark (same weights/optimizer/RNG snapshot, alternating arms)\n\n'+md(df[[c for c in cols if c in df.columns]]))
        corr=['case','arm','search_identity','rng_equal','logits_bitwise','loss_bitwise','root_scores_max_abs','log_prior_max_abs','grad_strict_failures','grad_max_abs','span_scorer_grad_norm']
        frag.append('### Correctness vs V3 eager reference (fixed weights)\n\n'+md(df[[c for c in corr if c in df.columns]].dropna(subset=['search_identity']) if 'search_identity' in df.columns else df[['case','arm']],'.3g'))
        comp=['case','arm','compile_first_call_ms','compile_second_call_ms','compiled_frames','graph_breaks','recompiles_during_timing','cudagraph_skips','cuda_graph_launches','step_median_ms']
        if 'compile_first_call_ms' in df.columns:frag.append('### Compile arms\n\n'+md(df[df.compile_first_call_ms.notna()][[c for c in comp if c in df.columns]]))
        ratios=[]
        for case in df.case.unique():
            s=json.loads((b/'performance'/'bf16'/case/'samples.json').read_text(encoding='utf-8'))
            for arm,v in s['paired_speedup_vs_reference'].items():ratios.append(dict(case=case,arm=arm,paired_ratio_median=v['median'],paired_ratio_min=v['minimum'],paired_ratio_max=v['maximum']))
        if ratios:frag.append('### Paired speed ratio V3/arm (per-sample pairs)\n\n'+md(pd.DataFrame(ratios),'.3f'))
    for f in sorted((b/'stream').glob('stream_mb*.csv')) if (b/'stream').exists() else []:
        frag.append(f'### Real consecutive micro-batch stream ({f.stem})\n\n'+md(pd.read_csv(f)))
    sd=root/'stage_d'
    if (sd/'cache_build_times.csv').exists():frag.append('### Stage D: geometry build times, reference vs vectorized (ms)\n\n'+md(pd.read_csv(sd/'cache_build_times.csv')))
    if (sd/'cache_stream_replay.csv').exists():frag.append('### Stage D: real stream replay by LRU capacity and builder\n\n'+md(pd.read_csv(sd/'cache_stream_replay.csv'),'.3f'))
    tr=root/'training'
    if (tr/'training_throughput.csv').exists():frag.append('### Short training comparison\n\n'+md(pd.read_csv(tr/'training_throughput.csv'),'.4g'))
    (out/'benchmark_tables.md').write_text('\n\n'.join(frag),encoding='utf-8');print('benchmark tables written')


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='reports/cky_v35');a=p.parse_args();root=Path(a.root);out=root/'tables';out.mkdir(parents=True,exist_ok=True)
    if (root/'audit'/'sweep.csv').exists():audit_tables(root,out)
    benchmark_tables(root,out)

if __name__=='__main__':main()
