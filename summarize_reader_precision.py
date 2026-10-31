"""Render the Reader isolation report from actual numerical/paired benchmark data."""
import argparse,json,math
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def table(headers,rows):return "\n".join(["| "+" | ".join(headers)+" |","|"+"|".join("---" for _ in headers)+"|"]+["| "+" | ".join(map(str,r))+" |" for r in rows])
def sci(x):return f"{float(x):.3g}"
def interval(x):return f"{x.step_median_ms:.1f} [{x.step_min_ms:.1f},{x.step_max_ms:.1f}]"


def main():
    p=argparse.ArgumentParser();p.add_argument("--output-dir",default="reports/cky_v25_reader");a=p.parse_args();root=Path(a.output_dir)
    read=lambda name:json.loads((root/name).read_text(encoding="utf-8"))
    cases=[read(f"case_b{b}.json") for b in (1,4)];bench=pd.read_csv(root/"benchmark.csv");dtype=read("dtype_b1.json")
    assert len(bench)==20 and bench.groupby("case").weight_sha256.nunique().eq(1).all()
    forward=[];grads=[];propagation=[];controls=[];dtypes=[];decoder_rows=[];profiler_rows=[]
    positions=["query_input","query_projection","key_input","key_projection","value_input","value_projection",
               "attention_logits","attention_probabilities","attention_aggregation_internal","attention_weighted_values",
               "output_projection","residual_output","return_to_decoder"]
    for f in ("B","R","E"):
        x=dtype[f][0]["4"]
        dtypes.append([f]+[x["tensors"][k]["actual_dtype"].replace("torch.","") for k in ("query_projection","key_projection","value_projection","attention_logits","attention_probabilities","attention_weighted_values","output_projection","return_to_decoder")])
    for c in cases:
        b=c["batch"]
        for f,z in c["cases"].items():
            for layer,v in z["forward"]["next_decoder_blocks"].items():
                decoder_rows.append([b,f,int(layer)+1,sci(v["max_absolute_error"]),sci(v["normalized_l2_error"]),sci(v["rms_error"]),str(v["actual_dtype"]),str(v["tensor_shape"])])
            ff=z["forward"];gs=z["gradient"]["global_statistics"]
            forward.append([b,f,sci(ff["root"]["semantic"]["normalized_l2_error"]),sci(ff["root"]["semantic"]["max_absolute_error"]),
                sci(ff["root"]["scores"]["max_absolute_error"]),sci(ff["root"]["log_prior"]["max_absolute_error"]),
                sci(ff["lm_logits"]["normalized_l2_error"]),sci(ff["lm_logits"]["max_absolute_error"]),sci(ff["loss"]["max_absolute_error"])])
            grads.append([b,f,sci(gs["gradient_norm_reference"]),sci(gs["gradient_norm_lazy"]),sci(gs["max_absolute_gradient_error"]),
                sci(gs["normalized_l2_gradient_error"]),z["gradient"]["strict_failed_parameter_count"],
                sum(q["nan"]+q["inf"] for q in z["gradient"]["nonfinite_gradients"].values())])
        for stage in positions:
            x,y=c["cases"]["B"]["forward"]["readers"]["4"][stage],c["cases"]["R"]["forward"]["readers"]["4"][stage]
            propagation.append([b,stage,sci(x["max_absolute_error"]),sci(y["max_absolute_error"]),sci(x["normalized_l2_error"]),sci(y["normalized_l2_error"]),sci(y["rms_error"])])
        for f,cc in c["controlled_input"].items():
            for item in cc:
                controls.append([b,f,item["identical_reader_memory"],sci(item["logits"]["max_absolute_error"]),
                    all(item["reader_entry_rng_equal"].values()),max(v["query_input"]["max_absolute_error"] for v in item["readers"].values())])
    module_rows=[]
    for c in cases:
        for f,z in c["cases"].items():
            for name,m in z["gradient"]["modules"].items():
                module_rows.append(dict(batch=c["batch"],family=f,module=name,norm_v2=m["grad_norm_reference"],norm_lazy=m["grad_norm_lazy"],
                    max_absolute=m["max_absolute_error"],relative_l2=m["normalized_l2_error"],significant_relative=m["significant_gradient_relative_error"]))
    pd.DataFrame(module_rows).to_csv(root/"gradient_modules.csv",index=False)
    module_table=[]
    for c in cases:
        for name,m in c["cases"]["B"]["gradient"]["modules"].items():
            r=c["cases"]["R"]["gradient"]["modules"][name]
            module_table.append([c["batch"],name,sci(m["normalized_l2_error"]),sci(r["normalized_l2_error"]),sci(m["max_absolute_error"]),sci(r["max_absolute_error"]),sci(r["significant_gradient_relative_error"])])
    performance=[];paired=[]
    for case,rows in bench.groupby("case",sort=False):
        sample=read(f"performance/{case}/samples.json");p=sample["paired"]
        profile_path=root/f"performance/{case}/profiler_timing.json"
        if profile_path.exists():
            for row in read(f"performance/{case}/profiler_timing.json")["records"]:
                profiler_rows.append([case,row["arm"],f"{row['profiled_update_wall_ms']:.1f}",f"{row['profiled_update_cuda_envelope_ms']:.1f}"])
        paired.append([case,f"{p['B_speedup']['median']:.3f}x",f"{p['R_speedup']['median']:.3f}x",
            f"{(p['R_over_B_lazy_cost']['median']-1)*100:+.2f}%",f"[{(p['R_over_B_lazy_cost']['minimum']-1)*100:+.2f}%,{(p['R_over_B_lazy_cost']['maximum']-1)*100:+.2f}%]"])
        for _,x in rows.iterrows():
            performance.append([case,x.arm,interval(x),f"{x.ce_tokens_per_second:.1f}",f"{x.peak_allocated_gib:.4f}",
                f"{x.reader_forward_median_ms:.2f}/{x.reader_backward_median_ms:.2f}",f"{x.forest_forward_median_ms:.1f}/{x.forest_backward_median_ms:.1f}",int(x.full_step_cudaLaunchKernel)])
    figures=root/"figures";figures.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(11,4),sharey=True)
    short=["K proj","V proj","logits","prob","weighted","out proj","residual","next block"]
    for ax,c in zip(axes,cases):
        for f in ("B","R"):
            d=c["cases"][f]["forward"];v=d["readers"]["4"]
            y=[v[k]["normalized_l2_error"] for k in ("key_projection","value_projection","attention_logits","attention_probabilities","attention_weighted_values","output_projection","residual_output")]+[d["next_decoder_blocks"]["4"]["normalized_l2_error"]]
            ax.plot(range(8),np.maximum(y,1e-12),"o-",label=f)
        ax.set_xticks(range(8),short,rotation=30,ha="right");ax.set_yscale("log");ax.set_title("main batch="+str(c["batch"]));ax.legend()
    axes[0].set_ylabel("Paired relative L2 (zeros: 1e-12)");fig.tight_layout();fig.savefig(figures/"reader_error_propagation.png",dpi=180);plt.close(fig)
    plt.figure(figsize=(6,4));pos=np.arange(2)
    for i,f in enumerate(("B","R","E")):
        plt.bar(pos+(i-1)*.22,[c["cases"][f]["gradient"]["global_statistics"]["normalized_l2_gradient_error"] for c in cases],.22,label=f)
    plt.xticks(pos,["B=1","B=4"]);plt.yscale("log");plt.ylabel("Full-model gradient relative L2");plt.legend();plt.tight_layout();plt.savefig(figures/"full_gradient_error.png",dpi=180);plt.close()
    plt.figure(figsize=(10,4));names=bench.case.drop_duplicates().tolist();pos=np.arange(len(names))
    for i,arm in enumerate(("B1","B2","B1-R","B2-R")):
        rs=bench[bench.arm==arm].set_index("case").loc[names];med=rs.step_median_ms.to_numpy()
        plt.bar(pos+(i-1.5)*.18,med,.18,label=arm,yerr=np.stack((med-rs.step_min_ms,rs.step_max_ms-med)),capsize=2)
    plt.xticks(pos,names,rotation=15,ha="right");plt.ylabel("Full update (ms)");plt.legend();plt.tight_layout();plt.savefig(figures/"full_update.png",dpi=180);plt.close()
    ratios=[c["root_invariance"]["improvement_ratio"] for c in cases]
    tests=(root/"logs/pytest.log").read_text(encoding="utf-8") if (root/"logs/pytest.log").exists() else "test log pending"
    test_line=next((x for x in tests.splitlines() if " passed" in x),"test log pending")
    text=f"""# CKY V2.5-R：Semantic Reader FP32 隔离实验

## 1. 实验目标和范围

判断 BF16 Reader 是否放大 V2/Lazy 的微小根差异，以及 FP32 Reader 能否改善完整梯度。**本轮结果接近情况 B：Reader 内部前向误差显著缩小，但全模型梯度仅有限改善。** 不更改正式配置，保留 BF16 Reader；FP32 配置保留作实验选项。没有执行20–50步 sanity、500步训练或 V3。

main_125m、forest_vector4、B=1/4、H=4、W=24、K=4、R=12；数值对照目标长度16，性能合成目标长度128，真实 batch 沿用原数据。每个数值 case 共享同一份权重、输入与 RNG，Dropout 始终开启（model.train）。CKY、探索/Delta、Backpointer、Lazy DAG、Composer、Scorer、Decoder 结构、参数数量、优化目标与学习率均不变。数值实验直接完整 CE backward，无优化器更新。主模型实际参数为148,840,709，Reader 策略不增减参数。

## 2. 实际 Precision Policy / Dtype 验证

B1/B2：BF16 outer，FP32 IEEE Composer，Reader 随外层 BF16 AMP；B1-R/B2-R：仅 Reader 增加 FP32 IEEE 区域；E1/E2：全 FP32 IEEE 正控制。源码入口 `models/precision_policy.py::PrecisionPolicy.reader`、`configure_model`，以及 `models/reader.py::SemanticReader.forward/_forward`。FP32 Reader 在禁用 AMP 的上下文内，将 Query、候选语义、系数、log-prior 转 FP32；QKV、attention、output、norm、gate、residual 均按原公式计算。返回时 cast 到原 hidden dtype；本模型该边界实测是 FP32，后续 Decoder Linear 仍由外层 BF16 决定。没有增添可训练参数，checkpoint 对旧缺失 reader 字段按 bf16 默认兼容，显式改变策略仍拒绝 resume。

{table(['Family','Q','K','V','logits','softmax','SDPA return','output proj','return Decoder'],dtypes)}

注意 **BF16 SDPA MATH 的 logits/softmax/内部 aggregation 本来就是 FP32**，而其 QKV 和加权结果返回为 BF16。不能把 B 描述为所有 attention 运算都 BF16。`dtype_trace.json` 保留实际 shape/dtype、bias dtype、native operator 序列和 backend policy。训练有可微 bias，沿用原 MATH 后端；`TorchDispatchMode` 只观察实际 `_safe_softmax`/bmm/dropout，未重算 attention，也未改变随机采样。单测验证观察器开启/关闭时 Reader 输出、所有输入/参数梯度及 RNG **逐位相同**。不是使用旧 `return_attention=True` 的手写诊断分支冒充 native SDPA。TF32 始终显式禁用。

## 3. Forward 数值对照

{table(['Batch','Family','root L2','root max abs','score max abs','prior max abs','LM logits L2','LM max abs','CE loss abs'],forward)}

B 与 R 的每个实现，其 root semantic、vector coefficient、log-prior **实际逐位不变**；候选与回溯完全一致。Reader 精度没有反馈到树或搜索。LM logits 相对 L2 在 batch1/4 均下降，但 CE 的单次绝对差并非单调改善：batch1 的 R loss 差反而大于 B；不把某个指标的降低外推为整体等价。原始全部 span selected-flat、retained Candidate ID、左右子节点 ID、根 ID 与 RNG 重放明细保存在 `search_b*_*.json` / `rng_b*.pt`（grad-enabled forward replay，只归档离散结构，不作为替换输入的 backward 证据）。

## 4. Reader 内部误差传播

首个 Reader（layer4）下表显示相同精度下的 V2/Lazy 配对误差；后续 Reader 的所有相同字段、shape、dtype 见 `case_b*.json`。

{table(['Batch','Position','B max abs','R max abs','B relative L2','R relative L2','R RMS'],propagation)}

![Reader propagation](figures/reader_error_propagation.png)

图中精确零值只为对数显示置于1e-12，原始 JSON 保留0。带 mask 的 attention logits 中匹配的 -inf 被标注为预期 masked infinities；范数/RMS 只在双方有限位置计算，任何不匹配非有限值单独计数，未把 masked -inf 作为模型溢出。

**实际观察：** FP32 Reader 显著减小 K/V 投影、attention 和 Reader residual 的配对差异。首个 Reader 的 Query 原本完全相同；batch1 的 Value Projection max abs 从3.90625e-3降到7.7486e-7，而 batch4 的 Key max abs 比 Value 更大。因此敏感点首先出现在低精度 K/V 投影及返回边界，不能把 Value 永久指定为所有输入的唯一最大误差源。

Reader 后第一个 Decoder Block（表中为1-based层号）：

{table(['Batch','Family','Decoder block','max abs','relative L2','RMS','dtype pair','shape'],decoder_rows)}

R 的早期 block 仍可保持很小误差，较深层/最终 logits 则重新出现明显差异；并非每个紧随 Reader 的 block 都立即大幅放大。全 FP32 正控制明显更一致。**推断：** BF16 下游的量化/反向路径仍是敏感区域；本轮没有证明具体 cuBLAS 算法或某个 Decoder 子算子是唯一根因。

相同 Reader 输入控制：只在 `torch.no_grad()` 中把 semantic、coefficient、log-prior、mask 替换为同一预缓存参考。Dropout 保持训练态，Query/Decoder 输入未人工替换。此时输出和后续 Query 的对照如下：

{table(['Batch','Family','shared memory','LM max abs','all entry RNG equal','max query input abs'],controls)}

缓存输入时，Reader 输出与最终 logits 实测完全一致，所有 Reader 入口 RNG 一致。它支持“正常差异由不完全相同的根输入触发”的解释，但 **不用于宣称完整模型 backward 等价**。该无梯度控制沿用原 automatic SDPA selector，未假设其后端与有梯度 MATH 相同；控制和正常 backward 数据分开保存。

## 5. 完整 Backward 梯度对照

{table(['Batch','Family','grad norm V2','grad norm Lazy','max abs','global relative L2','strict failed params','nonfinite grads'],grads)}

全模型梯度改善比 B/R：B=1 为 **{ratios[0]:.3f}x**，B=4 为 **{ratios[1]:.3f}x**。对应相对 L2 降低约 {100*(1-1/ratios[0]):.2f}% / {100*(1-1/ratios[1]):.2f}%，没有声称消除完整梯度差异。比值只在有效非零分母时计算，同时保留绝对误差；本轮 R 分母并不接近零。

{table(['Batch','Module','B grad L2','R grad L2','B max abs','R max abs','R significant relative'],module_table)}

`gradient_modules.csv` 对七个模块、三种精度、两种 batch 保留两侧完整梯度范数与全部五项误差指标（42行）。上表用于直接比较 B/R；例如 Reader/Composer/Syntax 均改善，但 Decoder 的主导梯度只小幅改善。vector head 是 span scorer 的子集，模块表有重叠，全局统计不重复计数。`gradient_comparison.json` 还保留逐参数两侧范数变化、原始最大相对误差、显著梯度误差（|grad|≥1e-5）、严格失败参数名与非有限梯度数。原 V2 容差不变：BF16 atol=.00015/rtol=.04；全 FP32 atol=3e-6/rtol=.0003。近零梯度不只靠逐元素 relative 判断。全局统计直接对全部独立参数计算，未重复计算 vector head 子集。**严格 BF16 梯度不匹配仍保留，不以放宽标准将其转成通过。**

![Full gradients](figures/full_gradient_error.png)

## 6. Benchmark 与显存成本

同设备、同轮次四臂 B1/B2/B1-R/B2-R，所有参数/AdamW/RNG 快照相同，均为 BF16 outer、FP32 IEEE Composer。每臂 warmup2、有效样本8，四臂旋转/反转顺序平衡位置。输入驻 GPU，正常 forward + CE backward + clip + AdamW，正确 CUDA 同步；setup/恢复、dtype audit、profiler 不计普通时间。沿用共同零梯度 AdamW 初始化 buffers 的快照协议，初始化中的 weight decay 对四臂相同。实际环境在 `benchmark_environment.json`，未与旧轮次数据直接计算任何比值。

{table(['Case','Arm','update median [min,max] ms','CE/s','peak GiB','Reader F/B ms','Forest F/B ms','full launches'],performance)}

同一个 repeat 的配对时间比，而非跨测量轮次：

{table(['Case','B V2/Lazy','R V2/Lazy','Lazy R/B extra cost','R/B sample range'],paired)}

![Full update](figures/full_update.png)

小幅负额外成本属于本轮采样波动，不能据此认定 FP32 Reader 更快。smoke W16 保留了最高2248.7ms的 B1-R 离群样本，未事后删除，其配对成本区间也很宽。其余 shape 的低额外成本只适用于本机这些 batch。Lazy 主模型 peak 基本相当，几 MiB 内差异不解释为稳定显存收益。

独立 profiler 补充重放（每臂单次，先 warmup2；相同参数快照 hash，**不用于任何加速比**）：

{table(['Case','Arm','profiled update wall ms','CUDA envelope ms'],profiler_rows)}

`performance/*/profiler_timing.json` 另存实际 worker IEEE backend。顶层 `benchmark_environment.json` 的 backend 是协调进程启动时状态，不能误读为 CUDA worker 的实际策略；worker 显式 configure_backend，再进行所有四臂实验。Profiler 计时只覆盖被 instrumentation 包裹的更新及同步，不含 profiler 初始化/收尾；常规耗时来自上面的8个无 profiler 样本。

Reader F/B 为三个独立 staged 诊断的 module CUDA event envelope 之和，包含 launch gaps，可能重叠；Reader 的 memory dict 不是 autograd 直接 Tensor 边界，backward hook envelope 不构成所有 Reader/上游 kernel 的严格分割。Forest backward 是既有 graph-cut 诊断，完整 update 是独立无 Hook 普通计时。组件不可相加还原整步，也不冒充纯 GPU active time。完整 median/min/max、实际 Q/K/V/logit/softmax/output dtype、NaN/Inf 检查在 CSV 和各 case audit JSON，CUDA launches 来自独立 profiler。NaN/Inf 输出数来自每臂独立 dtype audit forward；所有 timed/staged updates 的梯度均经 clip 的 error_if_nonfinite 检查，不把一次 forward audit 称为逐样本 logits 全量监控。预期 masked -inf 不计入输出/梯度非有限事件。

## 7. 是否需要局部 FP32 KV / Attention

本轮**未运行 R-KV/R-Attn**。全 Reader FP32 虽显著改善投影一致性，但完整梯度仅减少约7–9%，更接近情况 B，尚没有足够依据继续扩大/拆分精度实验。它们是用户定义的有条件可选项，不用未测方案填充结果；也不运行20–50步训练 sanity 或3 seeds×500步训练。

## 8. 最终推荐精度配置

依据本轮数值定位，**继续保留 V2.5-P 的 BF16 Reader**，除非后续存在独立训练稳定性收益证据。FP32 Reader 没有显著消除整个 BF16 Transformer 的梯度误差，不能因为 Value Projection 很一致便宣称预训练等价。即便成本较低，当前证据也不支持将其设为正式默认。

```yaml
precision:
  outer_autocast: bf16
  composer: fp32_ieee
  reader: bf16
  matmul_precision: ieee
  score_dtype: existing
```

`configs/precision/fp32_reader.yaml` 保留可复现的 FP32 Reader 实验入口，常规配置与默认 `score_first` 均未自动切换。所有原始数据/失败日志保留；本次实验只覆盖两种主模型 batch、一组初始化和这些输入，不将结果外推为所有长度/种子/训练阶段的结论。

## 9. 是否可以开始 V3

可以以已有 V2.5-P（FP32 IEEE Composer + BF16 Reader）作为固定精度基线，进入独立的 Score-only CKY Width-batched Tensorization 研究，继续严格检验分数加法/Top-K/探索/Mask/Backpointer。**本轮没有实现 V3，也没有通过进一步扩大 FP32 区域追求逐元素一致。**

复现（本机 py312 CUDA）：

```powershell
& 'E:/anaconda3/envs/py312/python.exe' validate_reader_precision.py
& 'E:/anaconda3/envs/py312/python.exe' benchmark_reader_precision.py --repeats 8
& 'E:/anaconda3/envs/py312/python.exe' audit_reader_search.py --batch 1
& 'E:/anaconda3/envs/py312/python.exe' audit_reader_search.py --batch 4
& 'E:/anaconda3/envs/py312/python.exe' audit_reader_profiler.py
& 'E:/anaconda3/envs/py312/python.exe' summarize_reader_precision.py
```

回归检查：`{test_line}`。数值诊断和性能脚本独立，避免在 benchmark 时并发 GPU 任务。与上一轮的差异位置是观察事实；“BF16 下游放大”和“更小精度边界可能有帮助”仅是相应证据下的解释/未测假说。
"""
    (root/"summary.md").write_text(text,encoding="utf-8")
    print("Reader report written",root/"summary.md")

if __name__=="__main__":main()
