"""Render the V2.5 report from measured CSV/JSON artifacts (standard library)."""
import csv
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent

def read(name): return json.loads((ROOT/name).read_text(encoding="utf-8"))
def rows(name): return list(csv.DictReader((ROOT/name).open(encoding="utf-8",newline="")))
def table(headers,values):
    return "\n".join(["| "+" | ".join(headers)+" |","|"+"|".join("---" for _ in headers)+"|"]+
                      ["| "+" | ".join(map(str,row))+" |" for row in values])
def number(x): return f"{float(x):.2f}"
def scientific(x): return f"{float(x):.3g}"
def interval(row,prefix): return f"{float(row[prefix+'_median_ms']):.1f} [{float(row[prefix+'_min_ms']):.1f}, {float(row[prefix+'_max_ms']):.1f}]"

def main():
    correct,grad,work=read("correctness.json"),read("gradient_comparison.json"),read("composer_workload.json")
    bench=rows("benchmark.csv"); ops=rows("operator_counts.csv")
    env=read("benchmark_environment.json")["environment"]
    pairs=[(a,b) for a,b in zip(bench[::2],bench[1::2])]
    fp32=rows("fp32/benchmark.csv") if (ROOT/"fp32/benchmark.csv").exists() else []
    cases={a["case"]:(a,b) for a,b in pairs}
    ma,mb=cases["main_b4_w24"]
    wa=work["main_b4_w24"]["lazy_semantic"]["audit"]
    assert wa["v2_composer_candidate_rows"]==43072
    assert wa["v25_required_node_occurrences"]<=1472
    assert wa["v25_actual_composer_rows"]==wa["v25_unique_dag_nodes"]<=wa["v25_required_node_occurrences"]
    gf=next(x for x in grad["cases"] if x["config"]=="main_125m" and x["precision"]=="fp32")
    gb=next(x for x in grad["cases"] if x["config"]=="main_125m" and x["precision"]=="bf16")
    groups=table(["模块","FP32 grad max abs","FP32 grad relative L2","BF16 grad max abs","BF16 grad relative L2"],
                 [[key,scientific(v["max_absolute_gradient_error"]),scientific(v["normalized_l2_gradient_error"]),
                   scientific(gb["groups"][key]["max_absolute_gradient_error"]),scientific(gb["groups"][key]["normalized_l2_gradient_error"])]
                  for key,v in gf["groups"].items()])
    training=table(["设备/精度","模式/steps","最大 loss 差","搜索身份始终相同","最终 scalar.bias 差","最终其他参数最大差"],
                   [[x["device"]+"/"+x["precision"],x["mode"]+"/"+str(x["steps"]),scientific(x["max_loss_error"]),
                     x["search_identities_unchanged"],scientific(x["trajectory"][-1]["neutral_scalar_bias_error"]),
                     scientific(x["trajectory"][-1]["max_non_gauge_parameter_error"])] for x in correct["training"]])
    perf=table(["case/精度","V2 update ms [min,max]","V2.5 update ms [min,max]","加速","CE/s V2→V2.5","peak allocated GiB V2→V2.5"],
               [[a["case"]+"/"+a["precision"],interval(a,"step"),interval(b,"step"),
                 f"{float(a['step_median_ms'])/float(b['step_median_ms']):.2f}x",
                 number(a["target_tokens_per_second"])+" → "+number(b["target_tokens_per_second"]),
                 f"{float(a['peak_allocated_gib']):.3f} → {float(b['peak_allocated_gib']):.3f}"]
                for a,b in pairs+([(fp32[0],fp32[1])] if fp32 else [])])
    phases=table(["case","Forest fwd ms V2→V2.5","Forest backward ms V2→V2.5","score search envelope ms","lazy semantic envelope ms","D2H blocking / CPU DAG / H2D host ms"],
                 [[a["case"],number(a["cky_forward_median_ms"])+" → "+number(b["cky_forward_median_ms"]),
                   number(a["cky_backward_median_ms"])+" → "+number(b["cky_backward_median_ms"]),
                   number(b["score_search_cuda_ms_median"]),number(b["semantic_cuda_ms_median"]),
                   " / ".join(number(b[k+"_median"]) for k in ("topology_d2h_blocking_ms","dag_cpu_ms","index_h2d_host_ms"))]
                  for a,b in pairs])
    workload=table(["case","V2 rows","root-tree internal occurrences","unique internal DAG / actual Composer rows","calls V2→V2.5","共享率"],
                   [[a["case"],a["composer_rows"],work[a["case"]]["lazy_semantic"]["audit"]["v25_required_node_occurrences"],
                     b["composer_rows"],a["composer_calls"]+" → "+b["composer_calls"],
                     f"{100*work[a['case']]['lazy_semantic']['audit']['sharing_ratio']:.1f}%"] for a,b in pairs])
    op=table(["case/arm","CKY stack / cat / launches","Full-step launches","forest / score autograd nodes","profiled device gaps total/max ms"],
             [[x["case"]+"/"+x["implementation"]," / ".join(x[k] for k in ("cky_forward_stack","cky_forward_cat","cky_forward_cudaLaunchKernel")),
               x["full_step_cudaLaunchKernel"],x["forest_autograd_nodes"]+" / "+x["score_autograd_nodes"],
               number(x["device_idle_gap_total_ms"])+" / "+number(x["device_idle_gap_max_ms"])]
              for x in ops if x["case"] in ("main_b4_w24","real_b2")])
    widths=", ".join(f"{x['width']}:{x['rows']}" for x in wa["per_width_active_rows"])
    matrix_errors={key:max(x["max_absolute_errors"].get(key,0) for x in correct["matrix"]+correct["edge_cases"])
                   for key in ("root_semantic","root_vector_coeff","root_scores","root_log_prior","Reader","LM logits","LM loss")}
    tests=(ROOT/"validation_logs/release_tests.log").read_text(encoding="utf-8") if (ROOT/"validation_logs/release_tests.log").exists() else "See test log"
    test_line=next((x for x in reversed(tests.splitlines()) if " passed" in x),"See test log")
    text=f"""# CKY V2.5 实验报告

本轮结论：**FP32 数学路径与梯度验证通过，并测到完整更新加速；主模型 BF16 未通过沿用 V2 的严格逐元素梯度容差，因此保留为实验选项，默认仍为 `score_first`。** 不把 Composer 行数下降视为全部验收通过。

## 1. 可行性审查

源码判定 Go（见 `feasibility.json`）。`ReferenceSemanticForest.forward` (`models/forest_reference.py:110–185`) 的 split 枚举、Top-R、两级探索、Top-K、Delta、Mask、可达性和根排序只依赖结构势、Mask 和随机数。`SpanScorer.forward` (`models/syntax.py:17`) 的 vector 输出不参与搜索；`SemanticReader.forward` (`models/reader.py:27`) 才将它拼入 Key。`LatentForestLM.forward` (`models/model.py:142,175`) 的训练目标是 LM CE，没有中间语义辅助损失或语义去重。

`Composer.forward` (`models/composer.py:16`) 是有序 Linear/GELU、确定性 Sigmoid 门控、左右残差和逐向量 LayerNorm；没有 Dropout、随机采样、BatchNorm 或运行统计。Decoder、Syntax 的 Dropout 保留。Composer、Decoder、Syntax、Reader 和主模型源码 SHA256 均与 V2 报告一致；没有新增可训练参数或 checkpoint 状态，没有改变优化器/数据/K/R。审查针对当前 Composer；已对训练态 Dropout/BatchNorm 加拒绝检查。

## 2. Score-only Chart

`ScoreCell` (`models/forest_lazy.py:18`) 只持有可微 score、mask、Candidate base 和左右子 Candidate ID。`LazyForestMixin.forward` (`:27,63–143`) 复刻 V2 的算术顺序与 RNG draw 顺序：先 `left + right`，再 `+ local`；Top-R priority detach，但 selected/retained score 不 detach。**采用方案 A，原标量分数图通过 `masked_log_softmax`、Reader log-prior 接受 LM 梯度。** 不重算、不 CPU 化结构势。

Candidate ID = span base + row × 该 span 的实际 slot 数 + slot，显式区分 batch/history、跨度与保留槽位。flat 解码使用实际叶子/欠满 beam 槽数；不假设固定 K²。debug spans 返回 score、mask、flat 身份和回溯指针，不补算中间语义或系数。`reference`、`score_first`、`lazy_semantic` 的 state_dict 完全兼容，原实现文件未被改写。

## 3. Lazy Semantic DAG Evaluation

`models/forest_lazy.py:158–221` 在根选择后，一次批量 D2H 读取整型回溯、Mask、根 ID 与长度。CPU 仅处理离散拓扑，按完整 Candidate ID memoize，从有效根反查闭包；检查合法子节点、同一 row、连续 split 与严格缩小的宽度，保证无环。它不按语义/分数去重。没有逐候选 GPU `.item()`/`.cpu()`/`.tolist()`，不会传输或重新创建无梯度的 score。

`models/forest_lazy.py:222–238` 将索引打包一次 long H2D，另上传根 emission mask。按宽度合并所有 B×H 的活跃节点，一个非空宽度只调用一次原 Composer；padding 与未达节点不进入 Composer。叶子复用独立 SemanticEmbedding，系数保持 `(uL + uR) + u_span`。共享节点仅计算一次，由 autograd 对所有父引用累加梯度；没有 detach 语义、冻结参数或清零真实梯度。空闭包且 W>1 时用零值参数 anchor 保留 V2 的零张量梯度/AdamW 可达性；W=1 不增加 Composer 梯度路径。

以下工作量中的 node 均指**内部节点**，叶子另计。B=4,H=4,n=24,K=4 的无共享上界是 `4×4×4×23=1472` 个 occurrence，实际为 {wa['v25_required_node_occurrences']}；共享后为 **{wa['v25_unique_dag_nodes']} 个内部 DAG 节点 / {wa['v25_actual_composer_rows']} Composer rows**，另有 {wa['v25_unique_leaf_nodes']} 个唯一叶子。V2 组合 {wa['v2_composer_candidate_rows']} 行，跳过 {wa['v25_unused_candidates_skipped']} 行，共享率 {wa['sharing_ratio']:.2%}。`unused_candidates_skipped` 包含 V2 组合后被 Top-K/Delta/根闭包丢弃的工作，不只是 DAG 去重收益。

{workload}

主模型 B4 每宽度实际行数（width:rows）：`{widths}`。拓扑 D2H {wa['topology_bytes']} bytes，紧凑索引+Mask H2D {wa['index_bytes']} bytes。全部 case 的每宽度分布、hook 实测输入形状和 CPU/D2H/H2D 原始时间见 `composer_workload.json` 及 `raw/*/composer_workload.json`；行数与调用数均由 Composer hook 交叉核验。

## 4. 正确性、随机性与梯度

发布检查：`{test_line}`。另见 `validation_logs`。矩阵实际完成 **{len(correct['matrix'])}** 例：B∈{{1,2,4}}，n∈{{1,4,8,16,24}}，K∈{{1,2,4}}，R∈{{4,12}}，scalar/vector，Eval/Train-explore，CPU FP32、CUDA FP32/BF16；还有 {len(correct['edge_cases'])} 个边界例。覆盖 Delta=0、并列分数、部分/全 -inf、空行、欠满 beam 和 padding。比较所有 span 的枚举 metadata、selected/retained flat、分数、mask、候选数；根树顺序与 RNG state 一致。额外主模型 FP32/BF16 前向在开启 Syntax/Reader/Decoder Dropout 时，CPU/CUDA RNG state 也完全一致，记录在 `full_model_rng.json`。单测另显式核查所有跨度的左右 Candidate ID、共享 DAG 梯度、空闭包零梯度和无界 K/R 的 Catalan 枚举。

矩阵/边界各输出最大绝对误差：`{matrix_errors}`。小模型矩阵使用 FP32 atol=3e-6/rtol=3e-5、BF16 atol=.004/rtol=.025；搜索身份、分数、Mask 和 log-prior 均要求精确一致。Reader、Decoder logits 和 LM loss 也逐例比较。

主模型 FP32 根语义 max abs={scientific(gf['forward_errors']['root_semantic'])}、relative L2={scientific(gf['forward_errors']['root_semantic_normalized_l2'])}；结构分数、系数、log-prior 误差为零。主模型 BF16 根语义 max abs={scientific(gb['forward_errors']['root_semantic'])}、relative L2={scientific(gb['forward_errors']['root_semantic_normalized_l2'])}；logits max abs={scientific(gb['forward_errors']['lm_logits_max_absolute_error'])}、relative L2={scientific(gb['forward_errors']['lm_logits_normalized_l2'])}，loss 差={scientific(gb['forward_errors']['lm_loss_max_absolute_error'])}。

BF16 主模型**原小模型逐元素 forward 容差未通过**，原始失败日志保留。为量化而补测的 forward 诊断采用 atol=.04/rtol=.04 且 relative L2≤.02；该口径不替代严格验收。所有梯度仍沿用 V2 的 FP32 atol=3e-6/rtol=.0003、BF16 atol=.00015/rtol=.04，并额外报告每模块 relative L2（BF16诊断界限 .05）。主模型 BF16 有 **{len(gb.get('violations',[]))}** 个参数未通过原逐元素梯度检查，整体标记 `{grad['status']}`，没有写成全部通过。

{groups}

`gradient_comparison.json` 包含 {len(grad['cases'])} 组 tiny/main 对照的逐参数与逐模块两侧梯度范数、max abs、max relative、显著梯度 relative 和 normalized L2。relative 的分母为 max(|ref|,|lazy|,1e-8)，另报告 |grad|≥1e-5 的误差；normalized L2 = ||g_lazy−g_ref||₂ / ||g_ref||₂。微小中性梯度的相对误差会很大，未隐藏。Span Scorer 的非零梯度证明 log-prior 路径保留；独立共享节点单测验证没有漏计或重复计算梯度。

{training}

训练逐步恢复相同 RNG，使用真实 AdamW/clip，不冻结、不修改或删除 scalar.bias。CPU 8 步身份保持；CUDA BF16 20 步的 loss、梯度范数、bias 梯度及参数偏差全部记录，**权重数值漂移后，搜索身份并非始终相同**。最大 loss 差约 0.0086；不能据此声称 BF16 长程训练等价。scalar.bias 为公共势方向：n 词树均有 2n−1 个势，偏置在句内 softmax 抵消。scalar 模式最后一层 feature LayerNorm bias 也产生公共势；其余漂移逐参数保留，不全部归为零空间。AdamW epsilon 会放大接近零的舍入梯度。根据 FP32 对照，BF16 batch/GEMM 与共享节点累加次序是数值差异的合理解释，但本轮未宣称 BF16 严格替换资格。

## 5. 成对性能

设备：{env['gpu']}，{env['memory_gib']:.1f} GiB，torch {env['torch']}，CUDA {env['cuda_build']}。每 case **两侧各 warmup 2、各采样 5**；每次恢复同一 model/AdamW/RNG CPU 快照并交替 arm 顺序。权重 SHA256 在 CSV 中逐行提供，恢复、audit、profiler 不计入普通测速。输入已驻 GPU；计时覆盖完整 forward、loss backward、clip 和一次 AdamW 更新。CE/s 只统计非 -100 的真实目标 token。没有测数据读取、验证、checkpoint 或多 microbatch 累积。

{perf}

七个 BF16 case 的完整更新中位数都有收益；额外 FP32 main B1 也有真实更新收益。**BF16 性能列仅说明运行速度，不消除第 4 节严格数值检查未通过的问题。** real_b2 使用真实缓存动态 batch，实际 H=4,W=20,T=17、CE=29（非合成 128×B token）；真实数据、batch 与 K/R 未改变。峰值为 `max_memory_allocated`，不混用 reserved/系统显存；main B1 BF16 峰值略增，不能宣称所有形状都省显存。

独立诊断阶段（ms，中位数；完整分布见 CSV）：

{phases}

`Forest fwd/backward` 来自同步的 staged graph-cut 诊断；score search / semantic 为另一轮 CUDA Event envelope。**它们不是纯 GPU kernel 时间，包含 CPU launch gaps，且不同诊断行不可相加或拿来替代普通完整更新计时。** D2H blocking 包括等候此前 GPU 工作；CPU DAG 单独计时；索引 H2D host 与 CUDA envelope 均在 CSV 提供 median/min/max。V2 未被强行重构为 score-only，独立 score-search 时间列留空，V2 保留完整 Forest forward/backward 对照。

{op}

Kernel 数按 CPU runtime `cudaLaunchKernel*` 调用统计；stack/cat 为原生算子次数。device gaps 来自独立 profiler 的合并 CUDA activity 区间（含拷贝），只覆盖首尾 activity 之间，受 profiler 扰动，不能当作普通运行的精确 GPU 利用率；完整计数与 >50us gap 数见 `operator_counts.csv`。本轮观察到 score autograd 图规模保持，语义图与完整 backward launch 减少。主模型 B4 的 score-search envelope 约 {float(mb['score_search_cuda_ms_median']):.1f}ms，lazy semantic 约 {float(mb['semantic_cuda_ms_median']):.1f}ms；D2H/CPU DAG/H2D host 约 {float(mb['topology_d2h_blocking_ms_median']):.2f}/{float(mb['dag_cpu_ms_median']):.2f}/{float(mb['index_h2d_host_ms_median']):.2f}ms。剩余瓶颈主要是保留的逐 split 标量搜索及其反向图、Python 调度与许多小算子，并非 Composer 行数本身；部分形状前向还变慢，但完整更新仍加速。

## 6. 结论与复现

源码可行性 Go；FP32 前向/梯度正确，完整 FP32 main B1 实测加速。主模型 B4 BF16 完整更新 {float(ma['step_median_ms']):.1f}→{float(mb['step_median_ms']):.1f}ms，约 {float(ma['step_median_ms'])/float(mb['step_median_ms']):.2f}x，内部 Composer rows 43072→{wa['v25_actual_composer_rows']}，但 BF16 严格检查不足，**没有完成全部严格验收，不更换默认实现**。本轮实现为可切换实验路径，保留 V2 与 Reference、原参数/优化器/数据和生产训练 tqdm。

PowerShell（本机 CUDA 环境）：

```powershell
$py312 = 'E:\\anaconda3\\envs\\py312\\python.exe'
& $py312 -m pytest -q --basetemp=.pytest_tmp_v25_reproduce -o cache_dir=.pytest_cache_v25_reproduce
& $py312 validate_cky_v25.py --output-dir reports/cky_v25
& $py312 benchmark_cky_v25.py --warmup 2 --repeats 5 --output-dir reports/cky_v25
& $py312 benchmark_cky_v25.py --precision fp32 --cases main_b1_w24 --warmup 2 --repeats 5 --output-dir reports/cky_v25/fp32
& $py312 train.py --config configs/smoke.yaml --forest-implementation lazy_semantic --output-dir runs/lazy
& $py312 reports/cky_v25/build_summary.py
```

全量 validator 会在写完诊断后对严格梯度不匹配返回非零退出码；`--cpu-only` 或 `--skip-main` 可缩小验证范围，不能据此推断主模型 BF16 通过。运行环境、每次 sample、staged 诊断、workload、所有模块梯度、四条连续训练轨迹及数值失败日志均保留。测速仅有五次样本、单卡和这些 batch，报告 min/max，不外推长训练收益。
"""
    # PowerShell paths use native single backslashes.
    text=text.replace(chr(92)*2,chr(92))
    (ROOT/"summary.md").write_text(text,encoding="utf-8")
    print("Wrote",ROOT/"summary.md")

if __name__=="__main__": main()
