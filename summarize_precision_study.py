"""Build the report and scientific plots entirely from recorded measurements."""
from __future__ import annotations
import argparse,json,math,re
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAMES={"A":"BF16","B":"FP32 IEEE","C":"FP32 TF32","D":"FP16 + scaler","E":"All FP32 IEEE","S":"FP32 score supplement"}
def table(headers,rows):
    return "\n".join(["| "+" | ".join(headers)+" |","|"+"|".join("---" for _ in headers)+"|"]+["| "+" | ".join(map(str,r))+" |" for r in rows])
def sci(x):return f"{float(x):.3g}"
def interval(r,p="step"):return f"{r[p+'_median_ms']:.1f} [{r[p+'_min_ms']:.1f},{r[p+'_max_ms']:.1f}]"


def main():
    p=argparse.ArgumentParser();p.add_argument("--output-dir",default="reports/cky_v25_precision");a=p.parse_args();root=Path(a.output_dir)
    read=lambda name:json.loads((root/name).read_text(encoding="utf-8"))
    env=read("run_environment.json")["environment"]
    numerical={g:read(f"numerical_{g}.json") for g in "ABCDES"}
    models={g:next(m for m in x["models"] if m["config"]=="main_125m") for g,x in numerical.items()}
    bench=pd.read_csv(root/"benchmark.csv");training=pd.read_csv(root/"stability.csv");vals=pd.read_csv(root/"stability_validation.csv")
    # Cross-precision samples must start from exactly matching weights per shape.
    assert len(bench)==70
    assert bench.groupby("case")["weight_sha256"].nunique().max()==1
    assert bench["nan_inf_count"].sum()==0 and bench["skipped_steps"].sum()==0
    figures=root/"figures";figures.mkdir(exist_ok=True)
    colors={g:c for g,c in zip("ABCDE",plt.get_cmap("tab10").colors)}
    def save(name):plt.tight_layout();plt.savefig(figures/(name+".png"),dpi=180);plt.close()
    plt.figure(figsize=(7,4))
    first=[]
    for g in "ABCDE":
        nodes=numerical[g]["fixed_trace"]["nodes"]
        heights=np.array([n["height"] for n in nodes]);values=np.array([max(n["errors"]["stored_state"]["normalized_l2_error"],1e-12) for n in nodes])
        plt.scatter(heights,values,color=colors[g],s=12,alpha=.6)
        unique=sorted(set(heights.tolist()))
        plt.plot(unique,[np.median(values[heights==h]) for h in unique],label=NAMES[g],color=colors[g])
        candidates=[]
        for n in sorted(nodes,key=lambda n:(n["height"],n["candidate_id"])):
            for component in ["left_input","right_input","gate","left_residual","right_residual","mlp.0","mlp.1","mlp.2","gate_sigmoid","residual","norm_input","norm","parent","stored_state"]:
                e=n["errors"].get(component,{})
                if e.get("max_absolute_error",0)>1e-5:candidates.append((n["height"],n["width"],component,e));break
        if candidates:
            h,w,c,e=candidates[0];first.append([g,h,w,c,sci(e["max_absolute_error"]),sci(e["normalized_l2_error"])])
        else:first.append([g,"—","—","none >1e-5","—","—"])
    plt.yscale("log");plt.xlabel("Subtree height (single fixed root closure)");plt.ylabel("State relative L2 (zeros: 1e-12)");plt.legend(fontsize=8);save("root_error_depth")
    modules=list(models["B"]["gradients"]);xx=np.arange(len(modules));width=.15
    plt.figure(figsize=(10,4))
    for i,g in enumerate("ABCDE"):
        plt.bar(xx+(i-2)*width,[max(models[g]["gradients"][k].get("normalized_l2_error",0),1e-12) for k in modules],width,label=NAMES[g],color=colors[g])
    plt.xticks(xx,modules,rotation=25,ha="right");plt.yscale("log");plt.ylabel("Gradient relative L2 (zeros: 1e-12)");plt.legend(fontsize=8);save("gradient_by_module")
    main_bench=bench[bench.case=="main_b4_w24"]
    plt.figure(figsize=(8,4))
    for j,impl in enumerate(["score_first","lazy_semantic"]):
        rows=main_bench[main_bench.implementation==impl].set_index("group").loc[list("ABCDE")]
        median=rows.step_median_ms.to_numpy();err=np.stack([median-rows.step_min_ms,rows.step_max_ms-median])
        plt.bar(np.arange(5)+(j-.5)*.35,median,.35,yerr=err,label=impl,capsize=3)
    plt.xticks(range(5),[NAMES[g] for g in "ABCDE"]);plt.ylabel("Full update (ms)");plt.legend();save("update_by_precision")
    plt.figure(figsize=(8,4))
    for arm,frame in training[(training.stage=="long")&(training.seed==42)].groupby("arm"):
        plt.plot(frame.step,frame.train_nll.rolling(20,min_periods=1).mean(),label=arm+" (20-update mean)")
    plt.xlabel("Successful optimizer update");plt.ylabel("Training NLL");plt.legend();save("training_loss")
    plt.figure(figsize=(8,4))
    for arm,frame in vals[vals.stage=="long"].groupby("arm"):
        stat=frame.groupby("step").agg(tokens=("target_tokens","mean"),nll=("val_nll","mean"),sd=("val_nll","std"))
        plt.plot(stat.tokens,stat.nll,"o-",label=arm)
        plt.fill_between(stat.tokens,(stat.nll-stat.sd).to_numpy(),(stat.nll+stat.sd).to_numpy(),alpha=.15)
    plt.xlabel("Actually consumed CE target tokens");plt.ylabel("Validation NLL (mean +/- seed SD)");plt.legend();save("validation_nll")
    diagnostics=[]
    for path in (root/"stability_runs").glob("*/*/state.json"):
        state=json.loads(path.read_text(encoding="utf-8"));diagnostics.append(dict(run=path.parent.name,stage=path.parent.parent.name,**state))
    long=[x for x in diagnostics if x["stage"]=="long"]
    health_rows=[]
    for (stage,arm),frame in training.groupby(["stage","arm"]):
        health_rows.append(dict(stage=stage,arm=arm,successful_rows=int((~frame.skipped_step.astype(bool)).sum()),
            skipped=int(frame.skipped_step.astype(bool).sum()),
            nonfinite_gradient_events=int(frame.nonfinite_gradient.fillna(False).astype(bool).sum()) if "nonfinite_gradient" in frame else 0,
            scale_min=float(frame.scaler_scale.min()),scale_max=float(frame.scaler_scale.max())))
    plt.figure(figsize=(10,4));pos=np.arange(len(health_rows))
    plt.bar(pos-.16,[x["skipped"] for x in health_rows],.32,label="Skipped attempts")
    plt.bar(pos+.16,[x["nonfinite_gradient_events"] for x in health_rows],.32,label="Recorded nonfinite-grad events")
    plt.xticks(pos,[x["stage"]+"/"+x["arm"] for x in health_rows],rotation=25,ha="right",fontsize=9)
    maximum=max(max(x["skipped"],x["nonfinite_gradient_events"]) for x in health_rows)
    plt.ylim(0,max(1,maximum*1.2))
    for q,x in enumerate(health_rows):plt.text(q,.025,str(x["skipped"])+" / "+str(x["nonfinite_gradient_events"]),ha="center",fontsize=8)
    plt.ylabel("Count");plt.legend(fontsize=8);save("scaler_skips")
    (root/"stability_diagnostics.json").write_text(json.dumps(health_rows,indent=2),encoding="utf-8")
    def perf(g,impl,case="main_b4_w24"):return bench[(bench.group==g)&(bench.implementation==impl)&(bench.case==case)].iloc[0]
    def global_gradient(g):
        groups=[v for k,v in models[g]["gradients"].items() if k!="vector_coefficient_head"]
        return math.sqrt(sum((x.get("grad_norm_reference",0)*x.get("normalized_l2_error",0))**2 for x in groups)/max(sum(x.get("grad_norm_reference",0)**2 for x in groups),1e-30))
    final=vals[(vals.stage=="long")&(vals.step==500)]
    decisions=[]
    for g in "ABCDE":
        for i,impl in enumerate(["score_first","lazy_semantic"],1):
            r=perf(g,impl);arm=g+str(i);v=final[final.arm==arm]
            vstr=f"{v.val_nll.mean():.5f} ± {v.val_nll.std(ddof=1):.5f} (n={len(v)})" if len(v) else "未入选/未运行"
            decisions.append([NAMES[g],impl,"Reference" if i==1 else sci(models[g]["forward"]["semantic"]["normalized_l2_error"]),
                "Reference" if i==1 else sci(global_gradient(g)),interval(r),f"{r.peak_allocated_gib:.3f}",vstr])
    perfrows=[]
    for g in "ABCDE":
        a0,b0=perf(g,"score_first"),perf(g,"lazy_semantic")
        perfrows.append([NAMES[g],f"{a0.step_median_ms:.1f} → {b0.step_median_ms:.1f}",f"{a0.step_median_ms/b0.step_median_ms:.2f}x",
            f"{a0.composer_forward_median_ms:.1f} → {b0.composer_forward_median_ms:.1f}",f"{a0.composer_backward_median_ms:.1f} → {b0.composer_backward_median_ms:.1f}",
            f"{a0.forest_forward_median_ms:.1f} → {b0.forest_forward_median_ms:.1f}",f"{a0.forest_backward_median_ms:.1f} → {b0.forest_backward_median_ms:.1f}",
            f"{a0.full_step_cudaLaunchKernel} → {b0.full_step_cudaLaunchKernel}"])
    speeds=[]
    for case in bench.case.drop_duplicates():
        speeds.append([case]+[f"{perf(g,'score_first',case).step_median_ms/perf(g,'lazy_semantic',case).step_median_ms:.2f}x" for g in "ABCDE"])
    gradrows=[[g,k,sci(v.get("grad_norm_reference",0)),sci(v.get("grad_norm_lazy",0)),sci(v.get("max_absolute_error",0)),sci(v.get("normalized_l2_error",0)),sci(v.get("significant_gradient_relative_error",0)),len(v.get("strict_bf16_failures",[]))] for g in "ABCDE" for k,v in models[g]["gradients"].items()]
    pd.DataFrame(gradrows,columns=["group","module","norm_ref","norm_lazy","max_abs","relative_l2","significant_relative","strict_bf16_parameter_failures"]).to_csv(root/"gradient_modules.csv",index=False)
    screens=vals[(vals.stage=="screen")&(vals.step==100)]
    screenrows=[[x.arm,f"{x.val_nll:.5f}",f"{x.val_perplexity:.3f}",f"{x.root_candidate_count:.3f}",f"{x.candidate_entropy:.3f}",sci(x.get("candidate_set_disagreement_vs_B1",0))] for _,x in screens.iterrows()]
    finalrows=[]
    for arm,x in final.groupby("arm"):
        ref=final[final.arm=="B1"].set_index("seed");paired=x.set_index("seed").val_nll-ref.val_nll
        tr=training[(training.stage=="long")&(training.arm==arm)]
        finalrows.append([arm,f"{x.val_nll.mean():.5f} ± {x.val_nll.std(ddof=1):.5f}",f"{paired.mean():+.5f}",f"{x.val_perplexity.mean():.3f}",
            f"{tr.step_seconds.mean()*1000:.1f}",f"{tr.cuda_envelope_ms.sum()/1000:.1f}",f"{tr.skipped_step.fillna(False).astype(bool).sum()}",
            f"{(tr.grad_norm>1).mean():.1%}",f"{x.candidate_set_disagreement_vs_B1.mean():.3f}"])
    score=read("score_precision_control.json");under=read("underflow_comparison.json");selection=read("selection.json")
    controls=read("controls_B.json");train_control=next(x for x in controls["cases"] if x["condition"]=="native_train" and x["comparison"]=="paired_impl")
    reader_detail={k:v for k,v in train_control["metrics"].items() if k.startswith("readers.4")}
    paired_seeds={arm:[float(x) for x in frame.sort_values("seed").val_nll] for arm,frame in final.groupby("arm")}
    assert all(len(v)==3 for v in paired_seeds.values())
    assert all(x["step"]==500 for x in long)
    mainB=perf("B","lazy_semantic");refB=perf("B","score_first")
    quality={}
    reference=final[final.arm=="B1"].set_index("seed").val_nll
    for arm,frame in final.groupby("arm"):
        if arm=="B1":continue
        delta=frame.set_index("seed").val_nll-reference
        quality[arm]=dict(mean_delta=float(delta.mean()),seeds_above_point1=int((delta>.1).sum()),
                          eligible=bool(delta.mean()<=.1 and (delta>.1).sum()<=1))
    eligible=[arm for arm,q in quality.items() if q["eligible"]]
    recommendation="B1"
    if eligible:
        recommendation=min(eligible,key=lambda arm:models[arm[0]]["forward"]["semantic"]["normalized_l2_error"])
        current=perf(recommendation[0],"lazy_semantic")
        for arm in eligible:
            other=perf(arm[0],"lazy_semantic")
            if other.step_max_ms<current.step_min_ms and other.step_median_ms<.95*current.step_median_ms:
                recommendation=arm;current=other
    recommendation_row=perf(recommendation[0],"score_first" if recommendation.endswith("1") else "lazy_semantic")
    decision=dict(selected=recommendation,quality=quality,
                  rule="Reject mean paired NLL degradation >.1 or >1/3 seeds above .1. Prefer root accuracy unless another eligible policy is >=5% faster with nonoverlapping five-sample ranges.")
    (root/"recommendation.json").write_text(json.dumps(decision,indent=2),encoding="utf-8")
    from precision_study import errors
    traceE=torch.load(root/"trace_values_E.pt",map_location="cpu",weights_only=False)
    cross=[]
    for g in "ABCD":
        t=torch.load(root/f"trace_values_{g}.pt",map_location="cpu",weights_only=False)
        cross.append([NAMES[g]]+[sci(errors(traceE["roots"][i],t["roots"][i])["normalized_l2_error"]) for i in range(2)])
    matrix_count=sum(len(x.get("matrix",{}).get("cases",[])) for x in numerical.values());edge_count=sum(len(x.get("matrix",{}).get("edge_cases",[])) for x in numerical.values())
    tests=(root/"logs/pytest.log").read_text(encoding="utf-8");test_line=next((x for x in tests.splitlines() if " passed" in x),"see pytest log")
    rootErrorTable=table(["Group","根 paired L2","根 max abs","最大模块 grad L2","严格 BF16 失败参数数"],[[g,sci(models[g]["forward"]["semantic"]["normalized_l2_error"]),sci(models[g]["forward"]["semantic"]["max_absolute_error"]),sci(max(x.get("normalized_l2_error",0) for x in models[g]["gradients"].values())),len({n for x in models[g]["gradients"].values() for n in x.get("strict_bf16_failures",[])})] for g in "ABCDE"])
    text=f"""# CKY V2.5-P：混合精度数值稳定性与性能实验

本轮完整执行数值矩阵、主模型逐层/梯度对照、35 组成对性能记录（70行、350个有效整步样本）、六种方案的 100 步筛选，以及入选方案 `{selection['selected']}` 的 500 步 × 3 seeds（13/42/2026）。默认 `score_first` 未更换；没有实现 V3。

## 1. 精度路径审查与实现

设备和版本见 `run_environment.json`、`feasibility.json`；实测是 {env['gpu']} / torch {env['torch']}、CUDA {env['cuda_build']}。参数均为 FP32 Master Weights，语义 Embedding 输出为真实 FP32。**原 BF16 AMP Composer 的 Linear/GELU/Gate/Residual 边界主要为 BF16，但 LayerNorm 输出已是 FP32，内部语义状态原本就保存为 FP32。** 因此不能把原误差归因于全程 BF16 状态。实际 Hook 记录在 `dtype_trace.json`，并非根据配置名称推断。

`models/precision_policy.py` 只改变上下文和必要 dtype 转换。B 在 Composer 内禁用 AMP，直接使用原 FP32 叶子与 FP32 参数，所有树状态保持 FP32；C 同样保留 FP32 Tensor，进程启动时允许 FP32 matmul 使用 TF32；非矩阵运算保持 FP32。D 内层 FP16 AMP，LayerNorm 自动输出 FP32，随后显式 cast 存储为 FP16，Master Weights 仍是 FP32，进入 Reader 前根 Tensor 因原叶子拼接提升至 FP32，Reader 投影仍走外层 BF16。临时 bank 的 FP32 叶子提升不等于丢失信息；所有 cast 均可微，没有 detach 训练路径。

每个精度组独立子进程设置一次 `torch.backends.fp32_precision='ieee'` 和 `torch.backends.cuda.matmul.fp32_precision='ieee'/'tf32'`，没有混用旧 `allow_tf32` setter。初始原环境为 `none`/`none`、旧 matmul string=`highest`；FP16/BF16 reduced-precision reduction 开关原为 True，保持不变，FP16 accumulation=False。C 的实际 Composer Linear 输出验证为 FP32，否则 benchmark 报错；完整 CUDA kernel 名称保存在各 case diagnostics JSON。允许 TF32 不代表每个小 GEMM 必然使用 Tensor Core，且该开关会影响进程中其他 FP32 matmul，跨精度不能将所有差异只归因于 Composer。API/舍入背景见 [PyTorch 2.12 numerical accuracy](https://docs.pytorch.org/docs/2.12/notes/numerical_accuracy.html)。

Composer 公式与 Decoder/Syntax/Scorer/Reader 架构、参数数量、K=4/R=12、探索/Delta 和优化目标保持。Composer 仅增加可选诊断观察现有中间结果。Checkpoint 保存并核查实际 PrecisionPolicy、config、GradScaler、optimizer/scheduler/sampler/RNG；恢复时拒绝策略不一致，补充了 CUDA FP16 恢复测试。

## 2. 固定权重、结构与梯度

A=BF16 Composer；B=FP32 IEEE Composer；C=FP32 TF32 Composer；D=FP16 Composer+scale；A–D 外层均为 BF16。E 是全 FP32 IEEE 正控制。

实际完成 {matrix_count} 个矩阵用例、{edge_count} 个边界例：B=1/2/4、W=4/8/16/24、K=1/2/4、R=4/12、Scalar/Vector、Eval/Train-explore；覆盖不可达、欠满、同分与 Padding。每个同精度配对的所有 span 候选身份、Backpointer、分数、Mask、根顺序和 RNG 一致。完整回归：`{test_line}`。另补充了 A–E 每组 B=4/W=24 的 smoke Scalar/Vector 完整模型 forward/backward 配对（共10组），见 `backward_batch4_*.json`；所有配对结构与 RNG 一致，覆盖更大的实际 Composer batch。IEEE E 的严格 FP32 梯度检查也保留。

{rootErrorTable}

严格 V2 梯度容差仍为 FP32 atol=3e-6/rtol=.0003、BF16 atol=.00015/rtol=.04，未放宽。`gradient_comparison.json` 与 `gradient_modules.csv` 同时保留各模块两侧范数、max abs、normalized L2、显著梯度 relative（|grad|≥1e-5），以及原严格检查失败列表。B 大幅改善根语义，却未使 BF16 Transformer 的所有逐元素梯度对齐；不能将这项检查写成全部通过。

![Gradient errors](figures/gradient_by_module.png)

## 3. 误差定位、独立 Score 与下溢控制

固定 IEEE FP32 叶子/势/系数/探索 RNG，只改变 Composer，按完整 Candidate ID 对齐一个样本的一棵根树闭包。每层保存 CPU 统计，不按 row index 强行配对；Linear、GELU、Gate、Residual、LayerNorm、parent 与存储状态的 max abs / relative L2 / RMS 全在 `layer_trace_*.json`。需要时用 `--export-width` 导出指定 width。首次超过 max-abs 1e-5 的观察如下；这是诊断阈值，不是语义正确性阈值。

{table(['Group','子树高度','Width','最早观察分量','max abs','relative L2'],first)}

![Error versus subtree height](figures/root_error_depth.png)

图中点为固定单树闭包的子树根状态，线为同一子树高度的中位数，不代表跨数据分布统计。对数图中的精确零值仅为显示放在1e-12，原始 JSON 保留0。叶子来自真实 FP32 源头。中间层 dtype/误差能定位差异出现与传播的位置，但仅凭 profiler kernel 名称不能确定具体 cuBLAS reduction 根因。

B 的相同实现重复对照 logits 差为零；模块入口 RNG 一致，Decoder 前四个 block 与第一个 Reader 的 query 输入一致，差异从 Reader 内部开始，详细投影/输出误差如下：

{table(['Reader operation','max abs','relative L2'],[[k,sci(v['max_absolute_error']),sci(v['normalized_l2_error'])] for k,v in reader_detail.items()])}

训练态下微扰随后在 BF16 Transformer 中放大；eval 对照本样本的 B 配对 logits 误差为零，强制 MATH SDPA 的训练对照仍有差异。证据支持下游量化放大，而非证明所有差异来自 Dropout、LayerNorm 或某个确定 GEMM 算法；完整控制在 `controls_A/B.json`。

局部 Scorer 输出/叶子为 BF16，width>=2 保留分数为 FP32，但第一次叶子相加发生在 `.float()` 之前。单独 score FP32 控制保持 Composer=B，只在进入 CKY 时 cast，保留加法/探索/Mask 顺序：原 leaf={score['existing_leaf_dtype']}，新 leaf={score['fp32_leaf_dtype']}，原 internal={score['existing_internal_dtype']}；改变的 selected/retained 身份条目 {score['changed_ids']}/{score['id_entries']}，根 score max abs={sci(score['root_score']['max_absolute_error'])}，log-prior relative L2={sci(score['log_prior']['normalized_l2_error'])}。这不会恢复已经量化的 scalar-head 输出，也不把结构切换归为 Composer 误差。补充 S 单独记录，固定结构诊断的势已经 FP32，故不重复。

Top-K margin 是实际 selected Top-R 池中 K/K+1 的分差，在探索进入 beam 与 Delta 之前记录，见各 `margins`；它只用于分析切换敏感性，不是语义指标。FP16 固定根探针的 scale-on/off 控制以及相对 E 非零梯度变为零的指标见 `underflow_*.json` / `underflow_comparison.json`：`{[dict(scaled=x['scaled'],zero_where_E_nonzero=x['zero_where_E_nonzero']) for x in under['results']]}`。零值差是下溢指示，不能排除导数本身舍入/消去；记录了未缩放 FP32 master gradient 的最小非零值、范围和非有限值。

## 4. 完整更新性能

每组每形状两侧 warmup≥2、有效样本=5，交替 V2/Lazy，每次恢复同一 Model/AdamW/RNG/GradScaler snapshot；表中 min/max 是原始样本范围。跨精度每个 case 的快照 SHA256 已核对相同。为了初始化同一 AdamW buffers，先执行共同的零梯度 AdamW setup（包含相同 weight decay），快照位于其后；setup、恢复、audit 与 profiler 不计普通计时。组件使用另三次 staged 诊断，Composer backward 为实际 full-backward-hook CUDA envelope 之和，含 launch gaps/可能重叠，组件不可相加还原普通整步。

{table(['Precision','Update ms V2→Lazy','加速','Composer F ms','Composer B ms','Forest F ms','Forest B ms','Full CUDA launches'],perfrows)}

{table(['Case']+list('ABCDE'),speeds)}

![Full update timing](figures/update_by_precision.png)

所有 70 行的 CE token/s、VRAM、rows/calls、CUDA launches、NaN/Inf、scale/skips、实际 dtype、median/min/max 都在 `benchmark.csv`。输入驻 GPU，包含正常 forward/backward/clip/AdamW，不包含数据读取/验证/保存。真实 real_b2 动态 W=20,T=17、CE=29。BF16 基线内部状态原已 FP32，所以 B 的**状态存储本身**无额外 FP32 成本；主要额外内存来自 Linear 激活/反向保存，peak 是整步实测而非仅状态字节。小 batch 的整步峰值可能由 Decoder/optimizer 决定。不能预设 TF32 更快；请按该表的样本范围判断，五个样本不足以证明几百分点差异可重复。

## 5. 短程训练与决策表

经用户选择，长实验使用原 smoke 模型（约 22.4M 参数），非 main_125m。训练子集为同一真实 TinyStories cache 中 104 个**完整缓存样本**，非空历史且每历史句≤12词、目标≤24 tokens，1137 CE tokens，未再截断。验证使用同一完整 dev：4 stories / 63 examples / 1041 CE tokens。所有策略 effective target batch=16、microbatch≤4；原 AdamW LR、betas、wd、clip=1、warmup=200、token schedule 不变。该局部数值研究不是大规模预训练质量证明。

{table(['Arm/100 steps','Val NLL','PPL','根数/历史','候选熵','集合分歧 vs B1'],screenrows)}

入选规则与所有候选数值在 `selection.json`：`{selection['rule']}`。入选 `{selection['selected']}`，每种均实际执行500次有效 optimizer update、3 seeds。seed42 从相同方案的100步 checkpoint 正确续训，计数为总共500而非再加500。FP16 scaling 按积累窗口保持常数，完成积累后 unscale，再 clip/step/update；skip 不计有效更新，规则依据 [PyTorch AMP examples](https://docs.pytorch.org/docs/2.12/notes/amp_examples.html)。

{table(['Arm/500','Val NLL mean±SD','paired ΔNLL vs B1','mean PPL','mean update ms','累计CUDA envelope s','skips','clip ratio','集合分歧 vs B1'],finalrows)}

![Training loss](figures/training_loss.png)

![Validation NLL](figures/validation_nll.png)

![Scaler skips](figures/scaler_skips.png)

曲线显示每20步训练 NLL 均值（seed42）和真实 token-weighted dev NLL；阴影是3个种子的标准差，不是置信区间。分歧以同一固定 probe 的树结构身份集合/顺序比较，结构分叉不自动等于训练失败。500步的 B1/B2 是同精度配对；C2 相对 B1 是整体方案比较，同时涉及精度与实现，不能把该 NLL 差单独归因于 Lazy 或 TF32。累计 CUDA Event envelope 包含设备等待/CPU launch 空隙，不冒充纯 kernel active time。训练时间包含真实数据取样与诊断范数；不能与驻 GPU benchmark 时间直接比较。

{table(['Precision','CKY','Root paired relative L2','Grad paired global L2','main B4 update ms [min,max]','peak GiB','500-step dev NLL'],decisions)}

## 6. 回答与建议

1. 首次显著树内误差位置见第3节逐层表；叶子没有先被转 BF16。原 LayerNorm 输出 FP32，但不能恢复前序低精度算子丢失的信息。
2. FP32 IEEE Composer 显著缩小固定权重根语义差异；完整 BF16 Transformer 梯度还受到下游数值放大，不宣称严格等价。
3. 相同固定树/输入、同一实现相对全 IEEE 的 root relative L2 如下；它与“同精度 V2/Lazy 配对误差”是不同问题，不声称 TF32 逐位一致。

{table(['Composer precision','V2 vs E1','Lazy vs E2'],cross)}

4. TF32 的整步速度实际见第4节，主要仍受 scalar CKY/调度/反向图影响，不预设 Tensor Core 一定更快。
5. 本样本 FP16 根配对误差低于 BF16，但不等于更适合训练；必须结合范围、scale 与验证结果。
6. 实际 NaN/Inf/skip 与 scale 见 CSV、health、underflow 和稳定性日志；不将跳过更新填成正常更新。
7. BF16 AMP 原本已存 FP32 节点，B 额外成本主要是神经算子保存项；实际 VRAM 表保留所有形状。
8. 本轮按验证退化门槛、根误差和实测速度选择 `{recommendation}`（规则与数据在 `recommendation.json`），不是预先认定 IEEE 最佳。B2 root paired L2={sci(models['B']['forward']['semantic']['normalized_l2_error'])}，main B4 {mainB.step_median_ms:.1f}ms，相对同精度 V2 {refB.step_median_ms/mainB.step_median_ms:.2f}x。若跨精度速度范围重叠，则优先更稳定的根精度，不把少量样本的几百分点差当成确定加速。
9. 500步逐种子 NLL 与配对差异：

{table(['Seed','B1 NLL','B2 NLL / delta','C2 NLL / delta'],[[seed,f"{reference.loc[seed]:.5f}"]+[f"{final[(final.arm==arm)&(final.seed==seed)].val_nll.iloc[0]:.5f} / {final[(final.arm==arm)&(final.seed==seed)].val_nll.iloc[0]-reference.loc[seed]:+.5f}" for arm in ['B2','C2']] for seed in [13,42,2026]])}

C2 在 seed42 相对 B1 的明显退化必须保留，不能因为 TF32 被允许就忽略该风险。筛选中的 .1 NLL 门槛是本轮操作性规则，不是非劣效统计检验。
三种子、小验证集只能支持局部稳定性判断，不能宣称可靠的大规模预训练结论或几百分之一 NLL 的普遍优势。训练 loss 下降而后期 dev NLL 上升与小子集过拟合一致；必须把共同的训练质量问题与精度方案之间的差别分开。
10. 本轮不自动更换现有 `score_first` 默认；建议以入选 `{recommendation}` 开展后续主模型预训练验证，再决定默认切换。严格数值失败记录仍保留。
11. 可以据此开始独立的 V3 研究，以固定的 B2/现有 score 路径作对照，继续严格检查结构/加法/探索顺序；本次未实现 V3。

## 7. 复现与限制

```powershell
& 'E:/anaconda3/envs/py312/python.exe' run_precision_study.py --resume
```

分阶段入口：`validate_precision_v25.py`、`benchmark_precision_v25.py`、`train_precision_stability.py`、`summarize_precision_study.py`。四份精度配置在 `configs/precision/`。Colab/其他 CUDA GPU 使用同一 CLI，并指定新的 `--output-dir`，如 `reports/cky_v25_precision_blackwell`；本报告只包含上述 GPU 的实测，不填其他 GPU 的数字。`PrecisionStudy_Colab.ipynb` 提供入口，`precision_study_cache.zip` 保留同一公共数据 cache。模型权重/optimizer/scaler checkpoints 在本机 stability_runs 里，源码包不收录大型 checkpoint 与完整模型原始梯度；保留了小型固定输入/逐层诊断 tensor 供重新生成图表。报告和失败日志均不删改原 V2/V2.5 检查。
"""
    (root/"summary.md").write_text(text,encoding="utf-8")
    pd.DataFrame(decisions,columns=["precision","cky","root_error","gradient_error","update_ms","peak_gib","val_nll_500"]).to_csv(root/"decision_matrix.csv",index=False)
    print("Report generated",root/"summary.md")

if __name__=="__main__":main()
