"""Assemble the top-level V3.5 deliverables from the stage outputs under reports/cky_v35/.

Produces (only from files that exist; nothing is synthesised):
  benchmark.csv            paired benchmark rows (both compile shape policies) + stream rows, with speed ratios vs the V3 eager arm
  compile_analysis.json    compile settings / cost / graph breaks / recompiles / CUDA-graph usage / amortisation / identity / training cost
  operator_counts.csv      kernel launches, autograd nodes, runtime sync calls per case x arm (+ audit rows)
  cache_analysis.json      audit cache statistics + Stage D build times and stream replay
  training_throughput.csv  Stage E per-arm summaries + steady-state / compile-cost columns derived from metrics.csv
"""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

ROOT=Path('reports/cky_v35')
# benchmark directory -> compile shape policy it was measured with
BENCH_DIRS=[('benchmark','static_rows_nontree_cudagraph'),('benchmark_dynamic_rows','dynamic_rows_cudagraph_trees')]


def load_json(p):
    p=Path(p);return json.loads(p.read_text(encoding='utf-8')) if p.exists() else None


def _bench_frame(d,config):
    b=ROOT/d/'benchmark.csv'
    if not b.exists():return None
    df=pd.read_csv(b);df['compile_shape_policy']=config;v3=df[df.arm=='v3'].set_index('case')
    df['v3_step_median_ms']=df.case.map(v3.step_median_ms);df['speedup_vs_v3_median']=df.v3_step_median_ms/df.step_median_ms
    df['v3_step_min_ms']=df.case.map(v3.step_min_ms);df['speedup_vs_v3_min']=df.v3_step_min_ms/df.step_min_ms
    paired=[]
    for case in df.case.unique():
        s=load_json(ROOT/d/'performance'/'bf16'/case/'samples.json')
        if not s:continue
        for arm,v in (s.get('paired_speedup_vs_reference') or {}).items():
            paired.append(dict(case=case,arm=arm,paired_ratio_median=v['median'],paired_ratio_min=v['minimum'],paired_ratio_max=v['maximum']))
    if paired:df=df.merge(pd.DataFrame(paired),on=['case','arm'],how='left')
    return df


def benchmark_csv():
    rows=[f for f in (_bench_frame(d,c) for d,c in BENCH_DIRS) if f is not None]
    if not rows:return None
    sd=ROOT/'benchmark'/'stream'
    for f in sorted(sd.glob('stream_mb*.csv')) if sd.exists() else []:
        s=pd.read_csv(f);s['case']=f'stream_mb{int(s.micro_batch_size.iloc[0])}_{int(s.steps.iloc[0])}steps';s['precision']='bf16';s['compile_shape_policy']=BENCH_DIRS[0][1]
        ref=s[s.arm=='v3']
        if len(ref):
            s['v3_total_wall_ms']=float(ref.total_wall_ms.iloc[0]);s['speedup_vs_v3_total_incl_compile']=s.v3_total_wall_ms/s.total_wall_ms
            s['speedup_vs_v3_last_quarter']=float(ref.last_quarter_median_ms.iloc[0])/s.last_quarter_median_ms
        rows.append(s)
    out=pd.concat(rows,ignore_index=True,sort=False);out.to_csv(ROOT/'benchmark.csv',index=False);return out


def _training_detail():
    """Per-arm steady-state and compile-cost figures from metrics.csv (Stage E)."""
    det={}
    for s in sorted((ROOT/'training').glob('*/summary.json')) if (ROOT/'training').exists() else []:
        arm=s.parent.name;m=pd.read_csv(s.parent/'metrics.csv');m=m[m.step>0]
        if not len(m):continue
        st=m.step_seconds;n=len(st);steady=st.iloc[n//2:]  # second half of the run: all common shapes seen, rare recompiles remain
        med=float(steady.median());spikes=m[(m.step_seconds>2*med)]
        det[arm]=dict(updates=n,step_seconds_median_second_half=med,step_seconds_median_last_quarter=float(st.iloc[3*n//4:].median()),
                      tokens_per_second_steady_median=4096/med,tokens_per_second_second_half_incl_recompiles=4096*len(steady)/float(steady.sum()),
                      first_update_seconds=float(st.iloc[0]),
                      compile_spike_steps=[int(x) for x in spikes.step],compile_spike_seconds=[round(float(x),1) for x in spikes.step_seconds],
                      compile_overhead_seconds_est=float((spikes.step_seconds-med).sum()),
                      val_nll_curve={int(r.step):float(r.val_nll) for _,r in m[m.val_nll.notna()].iterrows()},
                      train_nll_first3=[float(x) for x in m.train_nll.head(3)],grad_norm_first3=[float(x) for x in m.grad_norm.head(3)],
                      compiled_frames_recorded=int(m.compiled_frames.max()) if 'compiled_frames' in m else None)
    return det


def compile_analysis(bench):
    out=dict(description='torch.compile of the pure-tensor V3 width step (models/cky_compile.py). Costs are measured, never excluded from stream/training results.',
             final_settings=dict(fullgraph=True,fallback_random=True,emulate_precision_casts=True,recompile_limit=1024,dynamic='None (static by default)',
                                 rows_dim='symbolic via torch._dynamo.mark_dynamic on bank/mask_bank/span_flat/left_ids/right_ids',automatic_dynamic_shapes=False,
                                 reduce_overhead_cudagraph_trees=True,cudagraph_mark_step_begin='called before every compiled search forward',
                                 graphs_per_process='one graph per (words, width, explore/swap flags); rows symbolic; V3 with grad also compiles a backward graph per width'),
             first_settings_superseded=dict(dynamic=False,recompile_limit=512,reduce_overhead_cudagraph_trees=False,
                                            why_superseded='real training truncates the last micro-batch of every update, so rows change every update and each (rows, words, width) became a new static graph -> Dynamo recompile limit hit at update 6 (Stage E first attempt); non-tree CUDA graphs cannot hold a symbolic rows dim (ConstraintViolationError), hence trees + cudagraph_mark_step_begin',
                                            measured_in='reports/cky_v35/benchmark (8 cases) and benchmark/stream'))
    if bench is not None:
        c=bench[bench.arm.str.contains('compile',na=False)&bench.compile_first_call_ms.notna()] if 'compile_first_call_ms' in bench else pd.DataFrame()
        per=[]
        for _,r in c.iterrows():
            per.append(dict(compile_shape_policy=r.compile_shape_policy,case=r.case,arm=r.arm,compile_first_call_ms=r.compile_first_call_ms,second_call_ms=r.compile_second_call_ms,steady_step_median_ms=r.step_median_ms,
                            v3_step_median_ms=r.v3_step_median_ms,paired_ratio_median=r.get('paired_ratio_median'),per_step_saving_ms=r.v3_step_median_ms-r.step_median_ms,
                            break_even_micro_batch_steps=(r.compile_first_call_ms-r.compile_second_call_ms)/(r.v3_step_median_ms-r.step_median_ms) if r.v3_step_median_ms>r.step_median_ms else None,
                            graph_breaks=r.graph_breaks,recompiles_during_timing=r.recompiles_during_timing,cudagraph_skips=r.cudagraph_skips,cuda_graph_launches_per_step=r.cuda_graph_launches,
                            full_cuda_launches_per_step=r.full_cuda_launches,score_search_launches=r.score_search_launches,search_identity=r.get('search_identity'),grad_strict_failures=r.get('grad_strict_failures')))
        out['paired_cases']=per
        if per:
            first=[p['compile_first_call_ms'] for p in per];out['compile_first_call_ms_range']=[min(first),max(first)]
            out['note_first_call']='first-call cost depends on the on-disk inductor cache: the first compile arm of each case after a fresh TORCHINDUCTOR_CACHE_DIR pays the cold cost (~110-205 s on this CPU for all widths); later arms/cases reuse cached Triton kernels (4-60 s).'
        comp=[];pol=[p for _,p in BENCH_DIRS]
        for (case,arm),g in c.groupby(['case','arm']):
            if g.compile_shape_policy.nunique()==2:
                a=g.set_index('compile_shape_policy')
                comp.append(dict(case=case,arm=arm,static_paired_median=a.loc[pol[0],'paired_ratio_median'],dynamic_rows_paired_median=a.loc[pol[1],'paired_ratio_median'],
                                 static_step_median_ms=a.loc[pol[0],'step_median_ms'],dynamic_rows_step_median_ms=a.loc[pol[1],'step_median_ms'],
                                 static_search_fwd_ms=a.loc[pol[0],'score_forward_median_ms'],dynamic_rows_search_fwd_ms=a.loc[pol[1],'score_forward_median_ms'],
                                 cuda_graph_launches_equal=bool(a.loc[pol[0],'cuda_graph_launches']==a.loc[pol[1],'cuda_graph_launches'])))
        out['static_vs_dynamic_rows_rebenchmark']=comp
        st=bench[bench.case.str.startswith('stream',na=False)] if 'case' in bench else pd.DataFrame()
        out['stream']=[dict(case=r.case,arm=r.arm,steps=int(r.steps),unique_shapes=int(r.unique_shapes),compiled_frames=int(r.compiled_frames),fx_graphs_compiled=r.get('fx_graphs_compiled'),first_step_ms=r.first_step_ms,median_step_ms=r.median_step_ms,
                            last_quarter_median_ms=r.last_quarter_median_ms,total_wall_ms=r.total_wall_ms,speedup_total_incl_compile=r.get('speedup_vs_v3_total_incl_compile'),speedup_last_quarter=r.get('speedup_vs_v3_last_quarter'))
                       for _,r in st.iterrows()]
    ident=load_json(ROOT/'compile_identity.json')
    if ident:
        out['identity']={}
        for arm,v in ident['arms'].items():
            if isinstance(v,dict):
                flat=[]
                for k,cases in v.items():
                    if isinstance(cases,list):flat+=cases
                out['identity'][arm]=dict(cases=len(flat),all_pass=all(bool(c.get('pass',c.get('ok',True))) for c in flat if isinstance(c,dict)),graph_breaks=sum(int(c.get('graph_breaks',0) or 0) for c in flat if isinstance(c,dict)))
            else:out['identity'][arm]=v
        out['negative_control_no_emulate_precision_casts']=ident.get('negative_control_no_emulate_precision_casts')
    det=_training_detail()
    if det:
        out['training_stage_e']={arm:{k:v for k,v in d.items() if k not in ('val_nll_curve','train_nll_first3','grad_norm_first3')} for arm,d in det.items()}
        out['training_note']='compiled_frames in the Stage E metrics is 0 for the compile arms: the script read the absent "frames" Dynamo counter (fixed afterwards to stats.unique_graphs); the compile cost is visible instead as the first-update time and the recompile spikes (new words shapes).'
    (ROOT/'compile_analysis.json').write_text(json.dumps(out,indent=1,default=str),encoding='utf-8');return out


def operator_counts(bench):
    rows=[]
    if bench is not None:
        for _,r in bench[bench.case.notna()&~bench.case.str.startswith('stream',na=False)].iterrows():
            sync=r.runtime_sync_calls
            try:sync=eval(sync) if isinstance(sync,str) else {}
            except Exception:sync={}
            rows.append(dict(source='benchmark',compile_shape_policy=r.compile_shape_policy,case=r.case,arm=r.arm,batch=r.batch,words=r.words_actual,full_cuda_launches=r.full_cuda_launches,cky_forward_launches=r.cky_forward_launches,score_search_launches=r.score_search_launches,
                             cuda_graph_launches=r.cuda_graph_launches,score_autograd_nodes=r.score_autograd_nodes,forest_autograd_nodes=r.forest_autograd_nodes,loss_autograd_nodes=r.loss_autograd_nodes,reconstruction_nodes=r.reconstruction_nodes,
                             cudaMemcpyAsync=sync.get('cudaMemcpyAsync'),cudaStreamSynchronize=sync.get('cudaStreamSynchronize'),cudaDeviceSynchronize=sync.get('cudaDeviceSynchronize'),
                             gpu_busy_union_ms=r.gpu_busy_union_ms,gpu_idle_fraction=r.gpu_idle_fraction,profiled_wall_ms=r.profiled_wall_ms,step_median_ms=r.step_median_ms))
    d=ROOT/'audit'/'decomposition.csv'
    if d.exists():
        a=pd.read_csv(d)
        for _,r in a.iterrows():
            rows.append(dict(source='audit_v3_eager',case=f"real_mb{int(r.micro_batch_size)}",arm='v3',batch=r.micro_batch_size,full_cuda_launches=r.get('cuda_launches'),score_search_launches=r.get('launches_forest/search'),
                             score_autograd_nodes=r.get('graph_score_prior_autograd_nodes'),forest_autograd_nodes=r.get('graph_forest_output_autograd_nodes'),loss_autograd_nodes=r.get('graph_loss_autograd_nodes'),
                             gpu_busy_union_ms=r.get('profiled_gpu_busy_ms'),gpu_idle_fraction=r.get('profiled_gpu_idle_fraction'),step_median_ms=r.get('plain_wall_median_ms'),
                             launches_forward=r.get('launches_phase/forward'),launches_backward=r.get('launches_phase/backward'),launches_optimizer=r.get('launches_phase/optimizer'),launches_composer=r.get('launches_module/composer'),launches_decoder=r.get('launches_module/decoder')))
    df=pd.DataFrame(rows);df.to_csv(ROOT/'operator_counts.csv',index=False);return df


def cache_analysis():
    out=dict(description='Static CKY index-cache behaviour on the real probe stream (audit) and Stage D isolation measurements.')
    a=load_json(ROOT/'audit'/'cache_analysis.json')
    if a:
        out['audit']=dict(probe_seed=a['probe_seed'],stream_updates=a['stream_updates'],notes=a['notes'],
                          per_micro_batch={m:{k:v for k,v in d.items() if k!='key_frequency'}|dict(top_keys=d.get('key_frequency',[])[:6]) for m,d in a['per_micro_batch'].items() if m!='builds'})
        sim=ROOT/'audit'/'cache_simulation.csv'
        if sim.exists():out['audit']['lru_simulation']=pd.read_csv(sim).to_dict('records')
    for name,f in [('stage_d_build_times','cache_build_times.csv'),('stage_d_stream_replay','cache_stream_replay.csv')]:
        p=ROOT/'stage_d'/f
        if p.exists():out[name]=pd.read_csv(p).to_dict('records')
    o=load_json(ROOT/'stage_d'/'optimizer_check.json')
    if o:out['stage_d_optimizer_check_diagnostic_only']=o
    (ROOT/'cache_analysis.json').write_text(json.dumps(out,indent=1,default=str),encoding='utf-8');return out


def training():
    rows=[]
    for s in sorted((ROOT/'training').glob('*/summary.json')) if (ROOT/'training').exists() else []:
        r=load_json(s);r['run_dir']=s.parent.name;rows.append(r)
    if not rows:return None
    df=pd.DataFrame(rows);det=_training_detail()
    for col in ('step_seconds_median_second_half','step_seconds_median_last_quarter','tokens_per_second_steady_median','tokens_per_second_second_half_incl_recompiles','first_update_seconds','compile_overhead_seconds_est'):
        df[col]=df.run_dir.map(lambda a:det.get(a,{}).get(col))
    df['compile_spike_steps']=df.run_dir.map(lambda a:det.get(a,{}).get('compile_spike_steps'))
    if 'v3' in set(df.arm):
        ref=df[df.arm=='v3'].iloc[0];v3c=det.get('v3',{}).get('val_nll_curve',{})
        df['speedup_vs_v3_train_wall_incl_compile']=ref.train_wall_seconds/df.train_wall_seconds
        df['speedup_vs_v3_steady_median']=ref.step_seconds_median_second_half/df.step_seconds_median_second_half
        df['speedup_vs_v3_second_half_incl_recompiles']=df.tokens_per_second_second_half_incl_recompiles/ref.tokens_per_second_second_half_incl_recompiles
        df['break_even_updates_est']=[(c/(ref.step_seconds_median_second_half-s)) if (c and s and ref.step_seconds_median_second_half>s) else None for c,s in zip(df.compile_overhead_seconds_est,df.step_seconds_median_second_half)]
        df['val_nll_minus_v3']=df.final_val_nll-ref.final_val_nll
        df['val_nll_curve_max_abs_diff_vs_v3']=df.run_dir.map(lambda a:max((abs(v-v3c[k]) for k,v in det.get(a,{}).get('val_nll_curve',{}).items() if k in v3c),default=None))
        df['step1_train_nll_equal_v3']=df.run_dir.map(lambda a:(det.get(a,{}).get('train_nll_first3') or [None])[0]==(det.get('v3',{}).get('train_nll_first3') or [0])[0])
    df.to_csv(ROOT/'training_throughput.csv',index=False)
    curves=[]
    for m in sorted((ROOT/'training').glob('*/metrics.csv')):
        if m.parent.name.endswith('_static_attempt'):continue  # failed first attempt, kept as evidence only
        c=pd.read_csv(m);c['arm']=m.parent.name;curves.append(c)
    if curves:pd.concat(curves,ignore_index=True).to_csv(ROOT/'training'/'training_curves.csv',index=False)
    return df


if __name__=='__main__':
    b=benchmark_csv();compile_analysis(b);operator_counts(b);cache_analysis();t=training()
    print('benchmark rows',None if b is None else len(b),'| training runs',None if t is None else list(t.run_dir))
    if t is not None:
        pd.set_option('display.width',250)
        print(t[['run_dir','steps','train_wall_seconds','tokens_per_second_train_wall','step_seconds_median_second_half','tokens_per_second_steady_median','first_update_seconds','compile_overhead_seconds_est','compile_spike_steps','speedup_vs_v3_steady_median','speedup_vs_v3_second_half_incl_recompiles','break_even_updates_est','final_val_nll','val_nll_curve_max_abs_diff_vs_v3','step1_train_nll_equal_v3']].to_string())
