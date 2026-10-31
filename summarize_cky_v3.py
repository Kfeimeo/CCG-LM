"""Build the V3 report and scientific plots from saved experimental data."""
from pathlib import Path
import argparse,json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join('---' for _ in headers)+'|']+['| '+' | '.join(map(str,r))+' |' for r in rows])
def sci(x):return f'{x:.3g}'
def interval(x):return f'{x.step_median_ms:.1f} [{x.step_min_ms:.1f},{x.step_max_ms:.1f}]'

def main():
 p=argparse.ArgumentParser();p.add_argument('--output-dir',default='reports/cky_v3');a=p.parse_args();r=Path(a.output_dir)
 b=pd.read_csv(r/'benchmark.csv');w=pd.read_csv(r/'width_diagnostics.csv');mem=pd.DataFrame(read(r/'memory_profile.json'))
 g=read(r/'gradient_comparison.json');c=read(r/'correctness.json');assert len(b)==18 and len(c['matrix'])==1080
 result={};full=[];parts=[];ops=[];cache=[];memory=[];profile=[]
 for (precision,case),rows in b.groupby(['precision','case'],sort=False):
  x=rows[rows.implementation=='lazy_semantic'].iloc[0];y=rows[rows.implementation=='lazy_tensorized'].iloc[0];raw=read(r/'performance'/precision/case/'samples.json');audit=raw['arms']['lazy_tensorized']['workload']
  full.append([case,precision,interval(x),interval(y),f"{raw['paired_speedup']['median']:.3f}x",f'{x.ce_tokens_per_second:.1f} / {y.ce_tokens_per_second:.1f}'])
  parts.append([case,precision]+[f'{x[k]:.2f} / {y[k]:.2f}' for k in ['score_forward_median_ms','score_backward_median_ms','forest_forward_median_ms','forest_backward_median_ms','lazy_adapter_host_ms']])
  ops.append([case,precision]+[f'{int(x[k])} / {int(y[k])}' for k in ['score_autograd_nodes','cky_cudaLaunchKernel','full_cudaLaunchKernel','score_stack','score_cat','score_select']]+[f'{x.python_project_self_ms:.2f} / {y.python_project_self_ms:.2f}',f'{int(y.composer_rows)} / {int(y.composer_calls)}'])
  cache.append([case,precision,audit['index_cache_hit'],f"{audit['index_cache_hit_rate']:.3f}",f'{y.index_cache_total_build_host_ms:.2f}',f'{y.index_cache_bytes/2**20:.3f}'])
  mm=mem[(mem.case==case)&(mem.precision==precision)].set_index('implementation');mp=mm.loc['lazy_semantic'].peak_allocated_gib;mn=mm.loc['lazy_tensorized'].peak_allocated_gib
  memory.append([case,precision,f'{x.peak_allocated_gib:.4f} / {y.peak_allocated_gib:.4f}',f'{mp:.4f} / {mn:.4f}',f'{(mn-mp)*1024:+.2f}'])
  for impl in ('lazy_semantic','lazy_tensorized'):
   tt=raw['arms'][impl]['profiled_timing'];profile.append([case,precision,impl,f"{tt['wall_ms']:.1f}",f"{tt['cuda_envelope_ms']:.1f}"])
 result['full']=table(['Case','Precision','Lazy median [min,max] ms','V3 median [min,max] ms','paired ratio median','CE/s Lazy / V3'],full)
 result['parts']=table(['Case','Precision','Score F Lazy/V3 ms','Score B','Forest F','Forest B','DAG adapter host'],parts)
 result['ops']=table(['Case','Precision','score graph nodes','CKY launches','full launches','score stack','score cat','score select','Python self ms','Composer rows/calls'],ops)
 result['cache']=table(['Case','Precision','audit hit','cumulative hit rate','cold build host ms','cache MiB'],cache)
 result['memory']=table(['Case','Precision','shared-cache peak GiB','isolated active-cache peak GiB','isolated delta MiB'],memory)
 result['profile']=table(['Case','Precision','Arm','profiled wall ms','profiled CUDA envelope ms'],profile)
 grads=[];modules=[]
 for i,z in enumerate(g['cases']):
  gg=z['gradient'];q=gg['global_statistics'];grads.append([z['device'],z['precision'],'main' if z['main'] else z['mode'],z['batch'],sci(q['normalized_l2_gradient_error']),sci(q['max_absolute_gradient_error']),gg['strict_failed_parameter_count'],sum(t['nan']+t['inf'] for t in gg['nonfinite_gradients'].values())])
  for name,q in gg['modules'].items():
   if 'grad_norm_reference' in q:modules.append(dict(case=i,main=z['main'],precision=z['precision'],device=z['device'],batch=z['batch'],module=name,grad_norm_reference=q['grad_norm_reference'],grad_norm_v3=q['grad_norm_lazy'],max_absolute_gradient_error=q['max_absolute_error'],normalized_l2_gradient_error=q['normalized_l2_error'],significant_gradient_relative_error=q['significant_gradient_relative_error']))
 pd.DataFrame(modules).to_csv(r/'gradient_modules.csv',index=False)
 result['gradient']=table(['Device','Precision','Model/mode','Batch','global grad L2','max abs','strict failed params','nonfinite'],grads)
 result['modules']=table(['Precision','Batch','Module','grad norm Lazy','grad norm V3','max abs','relative L2','significant relative'],[[q['precision'],q['batch'],q['module']]+[sci(q[k]) for k in ['grad_norm_reference','grad_norm_v3','max_absolute_gradient_error','normalized_l2_gradient_error','significant_gradient_relative_error']] for q in modules if q['main']])
 fine=[]
 for (precision,case,impl),rows in w.groupby(['precision','case','implementation'],sort=False):
  ss=rows.groupby('repeat').sum(numeric_only=True);pair=ss.gather_pair_ms if impl=='lazy_semantic' else ss.gather_ms+ss.pair_ms
  fine.append([case,precision,impl,f'{pair.median():.2f}']+[f'{ss[k].median():.2f}' for k in ['selection_ms','decode_ms','assembly_ms']])
 result['fine']=table(['Case','Precision','Arm','Gather+Pair ms','Selection ms','ID decode ms','Chart assembly ms'],fine)
 selected=w[(w.case=='main_b4_w24')&(w.precision=='bf16')];per=[]
 for width,rows in selected.groupby('width'):
  x=rows[rows.implementation=='lazy_semantic'];y=rows[rows.implementation=='lazy_tensorized'];assert x.candidates.iloc[0]==y.candidates.iloc[0] and x.valid_candidates.iloc[0]==y.valid_candidates.iloc[0]
  per.append([width,int(y.candidates.iloc[0]),int(y.valid_candidates.iloc[0]),f'{x.elapsed_ms.median():.2f} / {y.elapsed_ms.median():.2f}',f'{int(x.kernel_launches.iloc[0])} / {int(y.kernel_launches.iloc[0])}'])
 result['width']=table(['Width','candidate capacity','valid candidates','envelope Lazy/V3 ms','launches Lazy/V3'],per)
 stages=[]
 for directory,label in [('stage_a','A / Lazy'),('stage_b_increment','B / A')]:
  frame=pd.read_csv(r/directory/'benchmark.csv')
  for case,rows in frame.groupby('case'):
   raw=read(r/directory/'performance/bf16'/case/'samples.json');q=raw['paired_speedup'];ref='lazy_semantic' if directory=='stage_a' else 'lazy_tensorized_a';dst='lazy_tensorized_a' if directory=='stage_a' else 'lazy_tensorized'
   x=rows[rows.implementation==ref].iloc[0];y=rows[rows.implementation==dst].iloc[0];stages.append([label,case,f'{x.step_median_ms:.2f} / {y.step_median_ms:.2f}',f"{q['median']:.3f}x",f"[{q['minimum']:.3f},{q['maximum']:.3f}]"])
 result['stages']=table(['Comparison','Case','same-round update ms','paired ratio median','sample ratio range'],stages)
 trains=[];trajectory=[]
 for filename in ['cpu_forest_scalar4.json','cpu_forest_vector4.json','cuda_real_training.json']:
  tt=read(r/'training'/filename)['trajectory'];trains.append([filename,len(tt)]+[sci(max(z[k] for z in tt)) for k in ['loss_error','max_parameter_error','neutral_bias_error']])
  trajectory.extend(dict(experiment=filename,step=z['step'],loss_reference=z['reference'].get('loss',z['reference'].get('train_nll')),loss_v3=z['v3'].get('loss',z['v3'].get('train_nll')),loss_error=z['loss_error'],max_parameter_error=z['max_parameter_error'],neutral_bias_error=z['neutral_bias_error']) for z in tt)
 pd.DataFrame(trajectory).to_csv(r/'training_trajectory.csv',index=False);result['training']=table(['Experiment','Steps','max loss difference','max parameter difference','max neutral-bias difference'],trains)
 actual=read(r/'training/cuda_real_training.json');acc=read(r/'accumulation_training/cuda_real_training.json')
 result.update(real_lazy=f"{actual['ce_tokens_per_second'][0]:.2f}",real_v3=f"{actual['ce_tokens_per_second'][1]:.2f}",acc_lazy=f"{acc['ce_tokens_per_second'][0]:.2f}",acc_v3=f"{acc['ce_tokens_per_second'][1]:.2f}",micro=str(sorted({z[k]['micro_batches'] for z in acc['trajectory'] for k in ('reference','v3')})))
 result['tests']=next(x for x in (r/'logs/pytest.log').read_text(encoding='utf-8').splitlines() if ' passed' in x)
 target=b[(b.case=='main_b4_w24')&(b.precision=='bf16')].set_index('implementation')
 x=target.loc['lazy_semantic'];y=target.loc['lazy_tensorized']
 for label,key in [('score_f','score_forward_median_ms'),('score_b','score_backward_median_ms'),('forest_f','forest_forward_median_ms'),('forest_b','forest_backward_median_ms')]:result[label]=f'{x[key]/y[key]:.2f}'
 result['update_speedup']=f"{read(r/'performance/bf16/main_b4_w24/samples.json')['paired_speedup']['median']:.3f}"
 result['kernel_reduction']=f"{100*(1-y.full_cudaLaunchKernel/x.full_cudaLaunchKernel):.2f}"
 figs=r/'figures';figs.mkdir(exist_ok=True);names=[f'{p}/{c}' for p,c in b[['precision','case']].drop_duplicates().itertuples(index=False,name=None)];pos=np.arange(len(names));fig,ax=plt.subplots(figsize=(10,5))
 for i,impl in enumerate(('lazy_semantic','lazy_tensorized')):
  rr=b[b.implementation==impl];v=rr.step_median_ms.to_numpy();ax.barh(pos+(i-.5)*.35,v,.35,label=impl,xerr=np.stack((v-rr.step_min_ms.to_numpy(),rr.step_max_ms.to_numpy()-v)),capsize=2)
 ax.set_yticks(pos,names);ax.invert_yaxis();ax.set_xlabel('Full microbatch update (ms), min/max bars');ax.legend();fig.tight_layout();fig.savefig(figs/'full_update.png',dpi=170);plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(10,4))
 for ax,prec in zip(axes,('bf16','fp32')):
  rr=b[(b.case=='main_b4_w24')&(b.precision==prec)].set_index('implementation');cols=['score_forward_median_ms','score_backward_median_ms','forest_forward_median_ms','forest_backward_median_ms']
  for i,impl in enumerate(('lazy_semantic','lazy_tensorized')):ax.bar(np.arange(4)+(i-.5)*.32,[rr.loc[impl,k] for k in cols],.32,label=impl)
  ax.set_xticks(range(4),['Score F','Score B','Forest F','Forest B']);ax.set_title('main B4 '+prec);ax.set_ylabel('Independent diagnostic median (ms)')
 handles,labels=axes[-1].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',ncol=2,fontsize=9)
 fig.tight_layout(rect=(0,0,1,.91));fig.savefig(figs/'main_components.png',dpi=170);plt.close(fig)
 fig,ax=plt.subplots(figsize=(8,4))
 for impl in ('lazy_semantic','lazy_tensorized'):
  rr=selected[selected.implementation==impl].groupby('width').elapsed_ms.median();ax.plot(rr.index,rr.values,'o-',label=impl)
 ax.set_xlabel('Span width');ax.set_ylabel('Diagnostic width envelope (ms)');ax.legend();fig.tight_layout();fig.savefig(figs/'width_envelope.png',dpi=170);plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(10,4));tt=pd.DataFrame(trajectory)
 for ax,experiment in zip(axes,('cpu_forest_vector4.json','cuda_real_training.json')):
  rr=tt[tt.experiment==experiment];ax.plot(rr.step,rr.loss_reference,'o-',label='Lazy');ax.plot(rr.step,rr.loss_v3,'x--',label='V3');ax.set_xlabel('Optimizer step');ax.set_ylabel('Training NLL');ax.set_title(experiment.replace('.json',''));ax.legend()
 fig.tight_layout();fig.savefig(figs/'training.png',dpi=170);plt.close(fig)
 template_path=r/'report_template.md'
 if not template_path.exists():template_path=Path(__file__).parent/'reports/cky_v3/report_template.md'
 template=template_path.read_text(encoding='utf-8');(r/'summary.md').write_text(template.format(**result),encoding='utf-8');print('V3 report written',r/'summary.md')

if __name__=='__main__':main()

