"""Figures for the V3.5 report (audit sweep, wall vs GPU-busy, benchmark arms, training curves)."""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

ROOT=Path('reports/cky_v35');FIG=ROOT/'figures';FIG.mkdir(parents=True,exist_ok=True)


def sweep():
    parts=[pd.read_csv(ROOT/'audit'/'sweep_part1_mb1-8.csv')]
    if (ROOT/'audit'/'part2'/'sweep.csv').exists():parts.append(pd.read_csv(ROOT/'audit'/'part2'/'sweep.csv'))
    s=pd.concat(parts,ignore_index=True);s=s[s.status=='ok']
    fig,ax=plt.subplots(1,2,figsize=(10,3.6))
    ax[0].plot(s.micro_batch_size,s.ce_tokens_per_second,'o-');ax[0].set_xscale('log',base=2);ax[0].set_xlabel('micro-batch size');ax[0].set_ylabel('CE tokens / s (4096-token update)')
    ax[0].set_title('Real-data update throughput (RTX 3070 Laptop 8GB)')
    for _,r in s.iterrows():
        if r.micro_batch_size>=16:ax[0].annotate('unreliable\n(shared-memory spill)',(r.micro_batch_size,r.ce_tokens_per_second),fontsize=7,textcoords='offset points',xytext=(5,5))
    ax[1].plot(s.micro_batch_size,s.update_wall_median_ms/1000,'o-',label='median');ax[1].plot(s.micro_batch_size,s.update_wall_min_ms/1000,'s--',label='min')
    ax[1].set_xscale('log',base=2);ax[1].set_yscale('log');ax[1].set_xlabel('micro-batch size');ax[1].set_ylabel('seconds per 4096-token update');ax[1].legend();ax[1].set_title('Update wall time')
    fig.tight_layout();fig.savefig(FIG/'sweep.png',dpi=130);plt.close(fig)


def decomposition():
    d=pd.read_csv(ROOT/'audit'/'decomposition.csv');g=d.groupby('micro_batch_size').median(numeric_only=True)
    fig,ax=plt.subplots(figsize=(7,3.6));x=range(len(g))
    ax.bar([i-0.2 for i in x],g.plain_wall_median_ms,width=0.4,label='uninstrumented step wall (ms)')
    ax.bar([i+0.2 for i in x],g.profiled_gpu_busy_ms,width=0.4,label='GPU kernel busy, union (ms, profiled run)')
    ax.set_xticks(list(x));ax.set_xticklabels([f'mb {int(i)}' for i in g.index]);ax.set_ylabel('ms per real micro-batch');ax.legend();ax.set_title('Step wall vs GPU busy: the step is CPU-launch bound')
    fig.tight_layout();fig.savefig(FIG/'wall_vs_busy.png',dpi=130);plt.close(fig)
    cols=['forest/search_cuda_ms','module/composer_cuda_ms','module/decoder_cuda_ms','module/reader_cuda_ms','module/syntax_cuda_ms','backward_cuda_ms','clip_cuda_ms','optimizer_cuda_ms']
    cols=[c for c in cols if c in g.columns]
    fig,ax=plt.subplots(figsize=(8,3.6));bottom=[0]*len(g)
    for c in cols:
        ax.bar(list(x),g[c],bottom=bottom,label=c.replace('_cuda_ms',''));bottom=[b+v for b,v in zip(bottom,g[c])]
    ax.set_xticks(list(x));ax.set_xticklabels([f'mb {int(i)}' for i in g.index]);ax.set_ylabel('CUDA event envelope ms');ax.legend(fontsize=7,ncol=2);ax.set_title('Stage envelopes per micro-batch (forest tail/D2H not shown)')
    fig.tight_layout();fig.savefig(FIG/'stage_envelopes.png',dpi=130);plt.close(fig)


def benchmark():
    b=ROOT/'benchmark'/'benchmark.csv'
    if not b.exists():return
    df=pd.read_csv(b);cases=list(dict.fromkeys(df.case));arms=list(dict.fromkeys(df.arm))
    fig,ax=plt.subplots(figsize=(max(8,1.2*len(cases)),4));w=0.8/len(arms)
    for k,arm in enumerate(arms):
        sub=df[df.arm==arm].set_index('case').reindex(cases)
        ax.bar([i+(k-len(arms)/2+.5)*w for i in range(len(cases))],sub.step_median_ms,width=w,label=arm)
    ax.set_xticks(range(len(cases)));ax.set_xticklabels(cases,rotation=30,ha='right',fontsize=8);ax.set_ylabel('full update median ms');ax.legend(fontsize=8);ax.set_title('Paired full-update time by arm')
    fig.tight_layout();fig.savefig(FIG/'benchmark_arms.png',dpi=130);plt.close(fig)


def training():
    t=ROOT/'training'/'training_curves.csv'
    if not t.exists():return
    c=pd.read_csv(t);fig,ax=plt.subplots(1,2,figsize=(10,3.6))
    for arm,g in c.groupby('arm'):
        v=g.dropna(subset=['val_nll'])
        ax[0].plot(v.target_tokens,v.val_nll,'o-',label=arm);ax[1].plot(v.train_seconds,v.val_nll,'o-',label=arm)
    ax[0].set_xlabel('CE target tokens');ax[0].set_ylabel('validation NLL');ax[0].legend();ax[1].set_xlabel('training wall seconds (sync, incl. CPU scheduling)');ax[1].set_ylabel('validation NLL');ax[1].legend()
    fig.tight_layout();fig.savefig(FIG/'training.png',dpi=130);plt.close(fig)


if __name__=='__main__':
    sweep();decomposition();benchmark();training();print('figures written')
