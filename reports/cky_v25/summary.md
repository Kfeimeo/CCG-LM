# CKY V2.5 实验报告

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

以下工作量中的 node 均指**内部节点**，叶子另计。B=4,H=4,n=24,K=4 的无共享上界是 `4×4×4×23=1472` 个 occurrence，实际为 1472；共享后为 **866 个内部 DAG 节点 / 866 Composer rows**，另有 384 个唯一叶子。V2 组合 43072 行，跳过 42206 行，共享率 41.17%。`unused_candidates_skipped` 包含 V2 组合后被 Top-K/Delta/根闭包丢弃的工作，不只是 DAG 去重收益。

| case | V2 rows | root-tree internal occurrences | unique internal DAG / actual Composer rows | calls V2→V2.5 | 共享率 |
|---|---|---|---|---|---|
| smoke_b2_w8 | 1312 | 224 | 169 | 7 → 7 | 24.6% |
| smoke_b2_w16 | 8352 | 480 | 242 | 15 → 15 | 49.6% |
| smoke_b2_w24 | 21536 | 736 | 294 | 23 → 17 | 60.1% |
| smoke_b8_w24 | 86144 | 2944 | 1615 | 23 → 23 | 45.1% |
| main_b1_w24 | 10768 | 368 | 230 | 23 → 23 | 37.5% |
| main_b4_w24 | 43072 | 1472 | 866 | 23 → 23 | 41.2% |
| real_b2 | 14176 | 404 | 223 | 19 → 19 | 44.8% |

主模型 B4 每宽度实际行数（width:rows）：`2:75, 3:20, 4:16, 5:22, 6:24, 7:26, 8:27, 9:30, 10:31, 11:34, 12:36, 13:40, 14:42, 15:42, 16:42, 17:38, 18:44, 19:48, 20:46, 21:51, 22:62, 23:6, 24:64`。拓扑 D2H 390912 bytes，紧凑索引+Mask H2D 21360 bytes。全部 case 的每宽度分布、hook 实测输入形状和 CPU/D2H/H2D 原始时间见 `composer_workload.json` 及 `raw/*/composer_workload.json`；行数与调用数均由 Composer hook 交叉核验。

## 4. 正确性、随机性与梯度

发布检查：`247 passed, 1 skipped in 53.51s`。另见 `validation_logs`。矩阵实际完成 **1080** 例：B∈{1,2,4}，n∈{1,4,8,16,24}，K∈{1,2,4}，R∈{4,12}，scalar/vector，Eval/Train-explore，CPU FP32、CUDA FP32/BF16；还有 48 个边界例。覆盖 Delta=0、并列分数、部分/全 -inf、空行、欠满 beam 和 padding。比较所有 span 的枚举 metadata、selected/retained flat、分数、mask、候选数；根树顺序与 RNG state 一致。额外主模型 FP32/BF16 前向在开启 Syntax/Reader/Decoder Dropout 时，CPU/CUDA RNG state 也完全一致，记录在 `full_model_rng.json`。单测另显式核查所有跨度的左右 Candidate ID、共享 DAG 梯度、空闭包零梯度和无界 K/R 的 Catalan 枚举。

矩阵/边界各输出最大绝对误差：`{'root_semantic': 1.1324882507324219e-06, 'root_vector_coeff': 0.0, 'root_scores': 0.0, 'root_log_prior': 0.0, 'Reader': 2.384185791015625e-07, 'LM logits': 1.9371509552001953e-07, 'LM loss': 9.5367431640625e-07}`。小模型矩阵使用 FP32 atol=3e-6/rtol=3e-5、BF16 atol=.004/rtol=.025；搜索身份、分数、Mask 和 log-prior 均要求精确一致。Reader、Decoder logits 和 LM loss 也逐例比较。

主模型 FP32 根语义 max abs=1.1e-06、relative L2=2.82e-07；结构分数、系数、log-prior 误差为零。主模型 BF16 根语义 max abs=0.0375、relative L2=0.00596；logits max abs=0.0156、relative L2=0.00647，loss 差=0.00119。

BF16 主模型**原小模型逐元素 forward 容差未通过**，原始失败日志保留。为量化而补测的 forward 诊断采用 atol=.04/rtol=.04 且 relative L2≤.02；该口径不替代严格验收。所有梯度仍沿用 V2 的 FP32 atol=3e-6/rtol=.0003、BF16 atol=.00015/rtol=.04，并额外报告每模块 relative L2（BF16诊断界限 .05）。主模型 BF16 有 **22** 个参数未通过原逐元素梯度检查，整体标记 `strict_tolerance_mismatch`，没有写成全部通过。

| 模块 | FP32 grad max abs | FP32 grad relative L2 | BF16 grad max abs | BF16 grad relative L2 |
|---|---|---|---|---|
| syntax_encoder | 1.73e-09 | 1.79e-06 | 1.24e-05 | 0.0127 |
| span_scorer | 2.98e-08 | 2.34e-06 | 6.68e-05 | 0.0138 |
| semantic_embedding | 2.04e-10 | 5.84e-07 | 3.58e-06 | 0.0086 |
| composer | 1.63e-09 | 5.96e-07 | 4.58e-05 | 0.0111 |
| vector_coefficient_head | 2.85e-09 | 8.47e-07 | 4.2e-05 | 0.0128 |
| semantic_reader | 1.63e-09 | 5.58e-07 | 5.13e-05 | 0.00853 |
| decoder | 3.58e-07 | 6.22e-07 | 0.00387 | 0.00871 |

`gradient_comparison.json` 包含 8 组 tiny/main 对照的逐参数与逐模块两侧梯度范数、max abs、max relative、显著梯度 relative 和 normalized L2。relative 的分母为 max(|ref|,|lazy|,1e-8)，另报告 |grad|≥1e-5 的误差；normalized L2 = ||g_lazy−g_ref||₂ / ||g_ref||₂。微小中性梯度的相对误差会很大，未隐藏。Span Scorer 的非零梯度证明 log-prior 路径保留；独立共享节点单测验证没有漏计或重复计算梯度。

| 设备/精度 | 模式/steps | 最大 loss 差 | 搜索身份始终相同 | 最终 scalar.bias 差 | 最终其他参数最大差 |
|---|---|---|---|---|---|
| cpu/fp32 | forest_scalar4/8 | 0 | True | 0.000517 | 0.000184 |
| cpu/fp32 | forest_vector4/8 | 4.77e-07 | True | 0.000123 | 2.79e-06 |
| cuda/bf16 | forest_scalar4/20 | 0.00774 | False | 0.0065 | 0.00782 |
| cuda/bf16 | forest_vector4/20 | 0.0086 | False | 0.000724 | 0.0071 |

训练逐步恢复相同 RNG，使用真实 AdamW/clip，不冻结、不修改或删除 scalar.bias。CPU 8 步身份保持；CUDA BF16 20 步的 loss、梯度范数、bias 梯度及参数偏差全部记录，**权重数值漂移后，搜索身份并非始终相同**。最大 loss 差约 0.0086；不能据此声称 BF16 长程训练等价。scalar.bias 为公共势方向：n 词树均有 2n−1 个势，偏置在句内 softmax 抵消。scalar 模式最后一层 feature LayerNorm bias 也产生公共势；其余漂移逐参数保留，不全部归为零空间。AdamW epsilon 会放大接近零的舍入梯度。根据 FP32 对照，BF16 batch/GEMM 与共享节点累加次序是数值差异的合理解释，但本轮未宣称 BF16 严格替换资格。

## 5. 成对性能

设备：NVIDIA GeForce RTX 3070 Laptop GPU，8.0 GiB，torch 2.12.1+cu132，CUDA 13.2。每 case **两侧各 warmup 2、各采样 5**；每次恢复同一 model/AdamW/RNG CPU 快照并交替 arm 顺序。权重 SHA256 在 CSV 中逐行提供，恢复、audit、profiler 不计入普通测速。输入已驻 GPU；计时覆盖完整 forward、loss backward、clip 和一次 AdamW 更新。CE/s 只统计非 -100 的真实目标 token。没有测数据读取、验证、checkpoint 或多 microbatch 累积。

| case/精度 | V2 update ms [min,max] | V2.5 update ms [min,max] | 加速 | CE/s V2→V2.5 | peak allocated GiB V2→V2.5 |
|---|---|---|---|---|---|
| smoke_b2_w8/bf16 | 167.5 [157.2, 182.4] | 146.8 [138.5, 149.2] | 1.14x | 1528.02 → 1743.75 | 0.413 → 0.413 |
| smoke_b2_w16/bf16 | 516.9 [490.1, 539.2] | 373.3 [368.2, 394.4] | 1.38x | 495.27 → 685.72 | 0.460 → 0.412 |
| smoke_b2_w24/bf16 | 1083.9 [1067.0, 1153.8] | 702.1 [663.7, 748.1] | 1.54x | 236.18 → 364.61 | 0.645 → 0.416 |
| smoke_b8_w24/bf16 | 1079.7 [1066.0, 1096.1] | 728.3 [725.4, 733.0] | 1.48x | 948.45 → 1405.96 | 1.682 → 0.650 |
| main_b1_w24/bf16 | 1165.5 [1150.9, 1206.3] | 823.3 [795.8, 846.2] | 1.42x | 109.83 → 155.47 | 2.719 → 2.724 |
| main_b4_w24/bf16 | 1177.7 [1157.3, 1203.9] | 834.7 [828.7, 866.8] | 1.41x | 434.75 → 613.37 | 2.945 → 2.742 |
| real_b2/bf16 | 736.6 [721.3, 777.8] | 564.1 [522.7, 578.2] | 1.31x | 39.37 → 51.41 | 0.512 → 0.407 |
| main_b1_w24/fp32 | 952.3 [938.6, 999.0] | 667.1 [644.1, 716.1] | 1.43x | 134.41 → 191.87 | 2.725 → 2.725 |

七个 BF16 case 的完整更新中位数都有收益；额外 FP32 main B1 也有真实更新收益。**BF16 性能列仅说明运行速度，不消除第 4 节严格数值检查未通过的问题。** real_b2 使用真实缓存动态 batch，实际 H=4,W=20,T=17、CE=29（非合成 128×B token）；真实数据、batch 与 K/R 未改变。峰值为 `max_memory_allocated`，不混用 reserved/系统显存；main B1 BF16 峰值略增，不能宣称所有形状都省显存。

独立诊断阶段（ms，中位数；完整分布见 CSV）：

| case | Forest fwd ms V2→V2.5 | Forest backward ms V2→V2.5 | score search envelope ms | lazy semantic envelope ms | D2H blocking / CPU DAG / H2D host ms |
|---|---|---|---|---|---|
| smoke_b2_w8 | 54.73 → 69.16 | 58.12 → 48.43 | 40.11 | 11.85 | 0.71 / 1.00 / 0.29 |
| smoke_b2_w16 | 158.06 → 165.39 | 268.66 → 163.63 | 112.79 | 24.35 | 1.06 / 1.43 / 0.27 |
| smoke_b2_w24 | 304.28 → 281.24 | 704.92 → 340.85 | 213.48 | 27.80 | 1.46 / 1.83 / 0.31 |
| smoke_b8_w24 | 313.05 → 321.09 | 676.28 → 372.43 | 218.87 | 38.10 | 2.58 / 9.31 / 0.64 |
| main_b1_w24 | 299.96 → 292.02 | 706.07 → 366.77 | 214.69 | 37.85 | 1.56 / 1.47 / 0.29 |
| main_b4_w24 | 301.30 → 291.26 | 702.22 → 363.98 | 226.38 | 41.12 | 2.00 / 5.56 / 0.53 |
| real_b2 | 227.06 → 229.81 | 445.02 → 246.87 | 163.43 | 30.92 | 1.21 / 1.62 / 0.26 |

`Forest fwd/backward` 来自同步的 staged graph-cut 诊断；score search / semantic 为另一轮 CUDA Event envelope。**它们不是纯 GPU kernel 时间，包含 CPU launch gaps，且不同诊断行不可相加或拿来替代普通完整更新计时。** D2H blocking 包括等候此前 GPU 工作；CPU DAG 单独计时；索引 H2D host 与 CUDA envelope 均在 CSV 提供 median/min/max。V2 未被强行重构为 score-only，独立 score-search 时间列留空，V2 保留完整 Forest forward/backward 对照。

| case/arm | CKY stack / cat / launches | Full-step launches | forest / score autograd nodes | profiled device gaps total/max ms |
|---|---|---|---|---|
| main_b4_w24/score_first | 1154 / 1309 / 4081 | 24241 | 5442 / 3032 | 1265.35 / 3.57 |
| main_b4_w24/lazy_semantic | 1130 / 1291 / 4093 | 14761 | 3791 / 3032 | 763.46 / 2.63 |
| real_b2/score_first | 802 / 929 / 3111 | 15488 | 3986 / 2140 | 2741.38 / 127.52 |
| real_b2/lazy_semantic | 782 / 915 / 3160 | 9726 | 2775 / 2140 | 495.64 / 1.81 |

Kernel 数按 CPU runtime `cudaLaunchKernel*` 调用统计；stack/cat 为原生算子次数。device gaps 来自独立 profiler 的合并 CUDA activity 区间（含拷贝），只覆盖首尾 activity 之间，受 profiler 扰动，不能当作普通运行的精确 GPU 利用率；完整计数与 >50us gap 数见 `operator_counts.csv`。本轮观察到 score autograd 图规模保持，语义图与完整 backward launch 减少。主模型 B4 的 score-search envelope 约 226.4ms，lazy semantic 约 41.1ms；D2H/CPU DAG/H2D host 约 2.00/5.56/0.53ms。剩余瓶颈主要是保留的逐 split 标量搜索及其反向图、Python 调度与许多小算子，并非 Composer 行数本身；部分形状前向还变慢，但完整更新仍加速。

## 6. 结论与复现

源码可行性 Go；FP32 前向/梯度正确，完整 FP32 main B1 实测加速。主模型 B4 BF16 完整更新 1177.7→834.7ms，约 1.41x，内部 Composer rows 43072→866，但 BF16 严格检查不足，**没有完成全部严格验收，不更换默认实现**。本轮实现为可切换实验路径，保留 V2 与 Reference、原参数/优化器/数据和生产训练 tqdm。

PowerShell（本机 CUDA 环境）：

```powershell
$py312 = 'E:\anaconda3\envs\py312\python.exe'
& $py312 -m pytest -q --basetemp=.pytest_tmp_v25_reproduce -o cache_dir=.pytest_cache_v25_reproduce
& $py312 validate_cky_v25.py --output-dir reports/cky_v25
& $py312 benchmark_cky_v25.py --warmup 2 --repeats 5 --output-dir reports/cky_v25
& $py312 benchmark_cky_v25.py --precision fp32 --cases main_b1_w24 --warmup 2 --repeats 5 --output-dir reports/cky_v25/fp32
& $py312 train.py --config configs/smoke.yaml --forest-implementation lazy_semantic --output-dir runs/lazy
& $py312 reports/cky_v25/build_summary.py
```

全量 validator 会在写完诊断后对严格梯度不匹配返回非零退出码；`--cpu-only` 或 `--skip-main` 可缩小验证范围，不能据此推断主模型 BF16 通过。运行环境、每次 sample、staged 诊断、workload、所有模块梯度、四条连续训练轨迹及数值失败日志均保留。测速仅有五次样本、单卡和这些 batch，报告 min/max，不外推长训练收益。
