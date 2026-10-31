# CKY V2.5-P：混合精度数值稳定性与性能实验

本轮完整执行数值矩阵、主模型逐层/梯度对照、35 组成对性能记录（70行、350个有效整步样本）、六种方案的 100 步筛选，以及入选方案 `['B1', 'B2', 'C2']` 的 500 步 × 3 seeds（13/42/2026）。默认 `score_first` 未更换；没有实现 V3。

## 1. 精度路径审查与实现

设备和版本见 `run_environment.json`、`feasibility.json`；实测是 NVIDIA GeForce RTX 3070 Laptop GPU / torch 2.12.1+cu132、CUDA 13.2。参数均为 FP32 Master Weights，语义 Embedding 输出为真实 FP32。**原 BF16 AMP Composer 的 Linear/GELU/Gate/Residual 边界主要为 BF16，但 LayerNorm 输出已是 FP32，内部语义状态原本就保存为 FP32。** 因此不能把原误差归因于全程 BF16 状态。实际 Hook 记录在 `dtype_trace.json`，并非根据配置名称推断。

`models/precision_policy.py` 只改变上下文和必要 dtype 转换。B 在 Composer 内禁用 AMP，直接使用原 FP32 叶子与 FP32 参数，所有树状态保持 FP32；C 同样保留 FP32 Tensor，进程启动时允许 FP32 matmul 使用 TF32；非矩阵运算保持 FP32。D 内层 FP16 AMP，LayerNorm 自动输出 FP32，随后显式 cast 存储为 FP16，Master Weights 仍是 FP32，进入 Reader 前根 Tensor 因原叶子拼接提升至 FP32，Reader 投影仍走外层 BF16。临时 bank 的 FP32 叶子提升不等于丢失信息；所有 cast 均可微，没有 detach 训练路径。

每个精度组独立子进程设置一次 `torch.backends.fp32_precision='ieee'` 和 `torch.backends.cuda.matmul.fp32_precision='ieee'/'tf32'`，没有混用旧 `allow_tf32` setter。初始原环境为 `none`/`none`、旧 matmul string=`highest`；FP16/BF16 reduced-precision reduction 开关原为 True，保持不变，FP16 accumulation=False。C 的实际 Composer Linear 输出验证为 FP32，否则 benchmark 报错；完整 CUDA kernel 名称保存在各 case diagnostics JSON。允许 TF32 不代表每个小 GEMM 必然使用 Tensor Core，且该开关会影响进程中其他 FP32 matmul，跨精度不能将所有差异只归因于 Composer。API/舍入背景见 [PyTorch 2.12 numerical accuracy](https://docs.pytorch.org/docs/2.12/notes/numerical_accuracy.html)。

Composer 公式与 Decoder/Syntax/Scorer/Reader 架构、参数数量、K=4/R=12、探索/Delta 和优化目标保持。Composer 仅增加可选诊断观察现有中间结果。Checkpoint 保存并核查实际 PrecisionPolicy、config、GradScaler、optimizer/scheduler/sampler/RNG；恢复时拒绝策略不一致，补充了 CUDA FP16 恢复测试。

## 2. 固定权重、结构与梯度

A=BF16 Composer；B=FP32 IEEE Composer；C=FP32 TF32 Composer；D=FP16 Composer+scale；A–D 外层均为 BF16。E 是全 FP32 IEEE 正控制。

实际完成 1440 个矩阵用例、50 个边界例：B=1/2/4、W=4/8/16/24、K=1/2/4、R=4/12、Scalar/Vector、Eval/Train-explore；覆盖不可达、欠满、同分与 Padding。每个同精度配对的所有 span 候选身份、Backpointer、分数、Mask、根顺序和 RNG 一致。完整回归：`282 passed, 1 skipped in 46.94s`。另补充了 A–E 每组 B=4/W=24 的 smoke Scalar/Vector 完整模型 forward/backward 配对（共10组），见 `backward_batch4_*.json`；所有配对结构与 RNG 一致，覆盖更大的实际 Composer batch。IEEE E 的严格 FP32 梯度检查也保留。

| Group | 根 paired L2 | 根 max abs | 最大模块 grad L2 | 严格 BF16 失败参数数 |
|---|---|---|---|---|
| A | 0.00596 | 0.0375 | 0.0138 | 22 |
| B | 2.95e-07 | 1.19e-06 | 0.0121 | 27 |
| C | 0.000463 | 0.00167 | 0.0141 | 26 |
| D | 0.000723 | 0.00391 | 0.0122 | 25 |
| E | 2.82e-07 | 1.1e-06 | 2.34e-06 | 0 |

严格 V2 梯度容差仍为 FP32 atol=3e-6/rtol=.0003、BF16 atol=.00015/rtol=.04，未放宽。`gradient_comparison.json` 与 `gradient_modules.csv` 同时保留各模块两侧范数、max abs、normalized L2、显著梯度 relative（|grad|≥1e-5），以及原严格检查失败列表。B 大幅改善根语义，却未使 BF16 Transformer 的所有逐元素梯度对齐；不能将这项检查写成全部通过。

![Gradient errors](figures/gradient_by_module.png)

## 3. 误差定位、独立 Score 与下溢控制

固定 IEEE FP32 叶子/势/系数/探索 RNG，只改变 Composer，按完整 Candidate ID 对齐一个样本的一棵根树闭包。每层保存 CPU 统计，不按 row index 强行配对；Linear、GELU、Gate、Residual、LayerNorm、parent 与存储状态的 max abs / relative L2 / RMS 全在 `layer_trace_*.json`。需要时用 `--export-width` 导出指定 width。首次超过 max-abs 1e-5 的观察如下；这是诊断阈值，不是语义正确性阈值。

| Group | 子树高度 | Width | 最早观察分量 | max abs | relative L2 |
|---|---|---|---|---|---|
| A | 1 | 2 | left_residual | 0.00391 | 0.00188 |
| B | — | — | none >1e-5 | — | — |
| C | 1 | 2 | left_residual | 0.000422 | 0.000274 |
| D | 1 | 2 | right_residual | 6.1e-05 | 6.45e-06 |
| E | — | — | none >1e-5 | — | — |

![Error versus subtree height](figures/root_error_depth.png)

图中点为固定单树闭包的子树根状态，线为同一子树高度的中位数，不代表跨数据分布统计。对数图中的精确零值仅为显示放在1e-12，原始 JSON 保留0。叶子来自真实 FP32 源头。中间层 dtype/误差能定位差异出现与传播的位置，但仅凭 profiler kernel 名称不能确定具体 cuBLAS reduction 根因。

B 的相同实现重复对照 logits 差为零；模块入口 RNG 一致，Decoder 前四个 block 与第一个 Reader 的 query 输入一致，差异从 Reader 内部开始，详细投影/输出误差如下：

| Reader operation | max abs | relative L2 |
|---|---|---|
| readers.4_input | 0 | 0 |
| readers.4.norm_input | 0 | 0 |
| readers.4.norm | 0 | 0 |
| readers.4.query_input | 0 | 0 |
| readers.4.query | 0 | 0 |
| readers.4.key_input | 1.19e-06 | 8.47e-08 |
| readers.4.key | 0.000488 | 2.36e-06 |
| readers.4.value_input | 1.19e-06 | 2.95e-07 |
| readers.4.value | 0.00391 | 0.000108 |
| readers.4.output_input | 0.00195 | 0.000117 |
| readers.4.output | 0.00195 | 0.000629 |
| readers.4 | 0.000244 | 1.56e-05 |

训练态下微扰随后在 BF16 Transformer 中放大；eval 对照本样本的 B 配对 logits 误差为零，强制 MATH SDPA 的训练对照仍有差异。证据支持下游量化放大，而非证明所有差异来自 Dropout、LayerNorm 或某个确定 GEMM 算法；完整控制在 `controls_A/B.json`。

局部 Scorer 输出/叶子为 BF16，width>=2 保留分数为 FP32，但第一次叶子相加发生在 `.float()` 之前。单独 score FP32 控制保持 Composer=B，只在进入 CKY 时 cast，保留加法/探索/Mask 顺序：原 leaf=torch.bfloat16，新 leaf=torch.float32，原 internal=torch.float32；改变的 selected/retained 身份条目 148/14732，根 score max abs=0.00226，log-prior relative L2=0.000249。这不会恢复已经量化的 scalar-head 输出，也不把结构切换归为 Composer 误差。补充 S 单独记录，固定结构诊断的势已经 FP32，故不重复。

Top-K margin 是实际 selected Top-R 池中 K/K+1 的分差，在探索进入 beam 与 Delta 之前记录，见各 `margins`；它只用于分析切换敏感性，不是语义指标。FP16 固定根探针的 scale-on/off 控制以及相对 E 非零梯度变为零的指标见 `underflow_*.json` / `underflow_comparison.json`：`[{'scaled': True, 'zero_where_E_nonzero': 49}, {'scaled': False, 'zero_where_E_nonzero': 3045}]`。零值差是下溢指示，不能排除导数本身舍入/消去；记录了未缩放 FP32 master gradient 的最小非零值、范围和非有限值。

## 4. 完整更新性能

每组每形状两侧 warmup≥2、有效样本=5，交替 V2/Lazy，每次恢复同一 Model/AdamW/RNG/GradScaler snapshot；表中 min/max 是原始样本范围。跨精度每个 case 的快照 SHA256 已核对相同。为了初始化同一 AdamW buffers，先执行共同的零梯度 AdamW setup（包含相同 weight decay），快照位于其后；setup、恢复、audit 与 profiler 不计普通计时。组件使用另三次 staged 诊断，Composer backward 为实际 full-backward-hook CUDA envelope 之和，含 launch gaps/可能重叠，组件不可相加还原普通整步。

| Precision | Update ms V2→Lazy | 加速 | Composer F ms | Composer B ms | Forest F ms | Forest B ms | Full CUDA launches |
|---|---|---|---|---|---|---|---|
| BF16 | 955.3 → 690.2 | 1.38x | 34.1 → 32.7 | 54.9 → 47.2 | 251.4 → 255.5 | 571.4 → 293.8 | 24241 → 14757 |
| FP32 IEEE | 928.4 → 655.6 | 1.42x | 29.2 → 28.0 | 53.4 → 40.0 | 240.6 → 252.2 | 570.9 → 285.6 | 23961 → 14908 |
| FP32 TF32 | 941.6 → 639.8 | 1.47x | 27.7 → 23.1 | 46.7 → 43.6 | 250.8 → 230.2 | 545.2 → 283.4 | 23679 → 14744 |
| FP16 + scaler | 1079.8 → 686.4 | 1.57x | 31.2 → 28.7 | 51.2 → 44.3 | 316.4 → 248.5 | 640.2 → 295.2 | 30724 → 15393 |
| All FP32 IEEE | 906.5 → 636.6 | 1.42x | 31.9 → 24.3 | 51.5 → 41.8 | 251.1 → 226.9 | 536.0 → 282.3 | 23429 → 14370 |

| Case | A | B | C | D | E |
|---|---|---|---|---|---|
| smoke_b2_w8 | 1.11x | 1.07x | 1.04x | 1.18x | 1.05x |
| smoke_b2_w16 | 1.30x | 1.36x | 1.34x | 1.45x | 1.43x |
| smoke_b2_w24 | 1.50x | 1.44x | 1.50x | 1.67x | 1.59x |
| smoke_b8_w24 | 1.45x | 1.48x | 1.43x | 1.65x | 1.40x |
| main_b1_w24 | 1.46x | 1.46x | 1.35x | 1.60x | 1.51x |
| main_b4_w24 | 1.38x | 1.42x | 1.47x | 1.57x | 1.42x |
| real_b2 | 1.36x | 1.38x | 1.51x | 1.60x | 1.48x |

![Full update timing](figures/update_by_precision.png)

所有 70 行的 CE token/s、VRAM、rows/calls、CUDA launches、NaN/Inf、scale/skips、实际 dtype、median/min/max 都在 `benchmark.csv`。输入驻 GPU，包含正常 forward/backward/clip/AdamW，不包含数据读取/验证/保存。真实 real_b2 动态 W=20,T=17、CE=29。BF16 基线内部状态原已 FP32，所以 B 的**状态存储本身**无额外 FP32 成本；主要额外内存来自 Linear 激活/反向保存，peak 是整步实测而非仅状态字节。小 batch 的整步峰值可能由 Decoder/optimizer 决定。不能预设 TF32 更快；请按该表的样本范围判断，五个样本不足以证明几百分点差异可重复。

## 5. 短程训练与决策表

经用户选择，长实验使用原 smoke 模型（约 22.4M 参数），非 main_125m。训练子集为同一真实 TinyStories cache 中 104 个**完整缓存样本**，非空历史且每历史句≤12词、目标≤24 tokens，1137 CE tokens，未再截断。验证使用同一完整 dev：4 stories / 63 examples / 1041 CE tokens。所有策略 effective target batch=16、microbatch≤4；原 AdamW LR、betas、wd、clip=1、warmup=200、token schedule 不变。该局部数值研究不是大规模预训练质量证明。

| Arm/100 steps | Val NLL | PPL | 根数/历史 | 候选熵 | 集合分歧 vs B1 |
|---|---|---|---|---|---|
| A1 | 7.42587 | 1678.867 | 3.943 | 1.035 | 0.45 |
| A2 | 7.42526 | 1677.829 | 3.925 | 0.538 | 0.35 |
| B1 | 7.42606 | 1679.176 | 3.925 | 0.658 | 0 |
| B2 | 7.42604 | 1679.147 | 3.943 | 0.540 | 0.3 |
| C2 | 7.42607 | 1679.201 | 3.943 | 1.143 | 0.5 |
| D2 | 7.42550 | 1678.247 | 3.943 | 1.084 | 0.35 |

入选规则与所有候选数值在 `selection.json`：`B1 reference + two finite lazy policies within .1 NLL of best, lowest paired root semantic L2 first (full-model gradient L2 <=.05 gate)`。入选 `['B1', 'B2', 'C2']`，每种均实际执行500次有效 optimizer update、3 seeds。seed42 从相同方案的100步 checkpoint 正确续训，计数为总共500而非再加500。FP16 scaling 按积累窗口保持常数，完成积累后 unscale，再 clip/step/update；skip 不计有效更新，规则依据 [PyTorch AMP examples](https://docs.pytorch.org/docs/2.12/notes/amp_examples.html)。

| Arm/500 | Val NLL mean±SD | paired ΔNLL vs B1 | mean PPL | mean update ms | 累计CUDA envelope s | skips | clip ratio | 集合分歧 vs B1 |
|---|---|---|---|---|---|---|---|---|
| B1 | 7.91951 ± 0.24052 | +0.00000 | 2805.486 | 217.1 | 325.5 | 0 | 100.0% | 0.000 |
| B2 | 7.90471 ± 0.18003 | -0.01481 | 2739.291 | 177.1 | 265.6 | 0 | 100.0% | 0.371 |
| C2 | 8.00605 ± 0.21999 | +0.08654 | 3046.696 | 181.8 | 272.6 | 0 | 100.0% | 0.384 |

![Training loss](figures/training_loss.png)

![Validation NLL](figures/validation_nll.png)

![Scaler skips](figures/scaler_skips.png)

曲线显示每20步训练 NLL 均值（seed42）和真实 token-weighted dev NLL；阴影是3个种子的标准差，不是置信区间。分歧以同一固定 probe 的树结构身份集合/顺序比较，结构分叉不自动等于训练失败。500步的 B1/B2 是同精度配对；C2 相对 B1 是整体方案比较，同时涉及精度与实现，不能把该 NLL 差单独归因于 Lazy 或 TF32。累计 CUDA Event envelope 包含设备等待/CPU launch 空隙，不冒充纯 kernel active time。训练时间包含真实数据取样与诊断范数；不能与驻 GPU benchmark 时间直接比较。

| Precision | CKY | Root paired relative L2 | Grad paired global L2 | main B4 update ms [min,max] | peak GiB | 500-step dev NLL |
|---|---|---|---|---|---|---|
| BF16 | score_first | Reference | Reference | 955.3 [900.9,983.4] | 2.944 | 未入选/未运行 |
| BF16 | lazy_semantic | 0.00596 | 0.00871 | 690.2 [649.0,724.0] | 2.734 | 未入选/未运行 |
| FP32 IEEE | score_first | Reference | Reference | 928.4 [906.8,980.5] | 3.106 | 7.91951 ± 0.24052 (n=3) |
| FP32 IEEE | lazy_semantic | 2.95e-07 | 0.00868 | 655.6 [640.8,678.6] | 2.734 | 7.90471 ± 0.18003 (n=3) |
| FP32 TF32 | score_first | Reference | Reference | 941.6 [884.2,967.6] | 3.106 | 未入选/未运行 |
| FP32 TF32 | lazy_semantic | 0.000463 | 0.00868 | 639.8 [616.2,666.3] | 2.734 | 8.00605 ± 0.21999 (n=3) |
| FP16 + scaler | score_first | Reference | Reference | 1079.8 [1061.4,1160.2] | 2.879 | 未入选/未运行 |
| FP16 + scaler | lazy_semantic | 0.000723 | 0.00865 | 686.4 [680.2,740.0] | 2.734 | 未入选/未运行 |
| All FP32 IEEE | score_first | Reference | Reference | 906.5 [873.3,935.0] | 3.140 | 未入选/未运行 |
| All FP32 IEEE | lazy_semantic | 2.82e-07 | 6.22e-07 | 636.6 [601.7,641.0] | 2.754 | 未入选/未运行 |

## 6. 回答与建议

1. 首次显著树内误差位置见第3节逐层表；叶子没有先被转 BF16。原 LayerNorm 输出 FP32，但不能恢复前序低精度算子丢失的信息。
2. FP32 IEEE Composer 显著缩小固定权重根语义差异；完整 BF16 Transformer 梯度还受到下游数值放大，不宣称严格等价。
3. 相同固定树/输入、同一实现相对全 IEEE 的 root relative L2 如下；它与“同精度 V2/Lazy 配对误差”是不同问题，不声称 TF32 逐位一致。

| Composer precision | V2 vs E1 | Lazy vs E2 |
|---|---|---|
| BF16 | 0.00589 | 0.00587 |
| FP32 IEEE | 0 | 0 |
| FP32 TF32 | 0.000459 | 7.48e-05 |
| FP16 + scaler | 0.000764 | 0.000786 |

4. TF32 的整步速度实际见第4节，主要仍受 scalar CKY/调度/反向图影响，不预设 Tensor Core 一定更快。
5. 本样本 FP16 根配对误差低于 BF16，但不等于更适合训练；必须结合范围、scale 与验证结果。
6. 实际 NaN/Inf/skip 与 scale 见 CSV、health、underflow 和稳定性日志；不将跳过更新填成正常更新。
7. BF16 AMP 原本已存 FP32 节点，B 额外成本主要是神经算子保存项；实际 VRAM 表保留所有形状。
8. 本轮按验证退化门槛、根误差和实测速度选择 `B2`（规则与数据在 `recommendation.json`），不是预先认定 IEEE 最佳。B2 root paired L2=2.95e-07，main B4 655.6ms，相对同精度 V2 1.42x。若跨精度速度范围重叠，则优先更稳定的根精度，不把少量样本的几百分点差当成确定加速。
9. 500步逐种子 NLL 与配对差异：

| Seed | B1 NLL | B2 NLL / delta | C2 NLL / delta |
|---|---|---|---|
| 13 | 7.72352 | 7.72202 / -0.00150 | 7.76903 / +0.04551 |
| 42 | 7.84710 | 7.91017 / +0.06306 | 8.20368 / +0.35658 |
| 2026 | 8.18793 | 8.08195 / -0.10598 | 8.04545 / -0.14248 |

C2 在 seed42 相对 B1 的明显退化必须保留，不能因为 TF32 被允许就忽略该风险。筛选中的 .1 NLL 门槛是本轮操作性规则，不是非劣效统计检验。
三种子、小验证集只能支持局部稳定性判断，不能宣称可靠的大规模预训练结论或几百分之一 NLL 的普遍优势。训练 loss 下降而后期 dev NLL 上升与小子集过拟合一致；必须把共同的训练质量问题与精度方案之间的差别分开。
10. 本轮不自动更换现有 `score_first` 默认；建议以入选 `B2` 开展后续主模型预训练验证，再决定默认切换。严格数值失败记录仍保留。
11. 可以据此开始独立的 V3 研究，以固定的 B2/现有 score 路径作对照，继续严格检查结构/加法/探索顺序；本次未实现 V3。

## 7. 复现与限制

```powershell
& 'E:/anaconda3/envs/py312/python.exe' run_precision_study.py --resume
```

分阶段入口：`validate_precision_v25.py`、`benchmark_precision_v25.py`、`train_precision_stability.py`、`summarize_precision_study.py`。四份精度配置在 `configs/precision/`。Colab/其他 CUDA GPU 使用同一 CLI，并指定新的 `--output-dir`，如 `reports/cky_v25_precision_blackwell`；本报告只包含上述 GPU 的实测，不填其他 GPU 的数字。`PrecisionStudy_Colab.ipynb` 提供入口，`precision_study_cache.zip` 保留同一公共数据 cache。模型权重/optimizer/scaler checkpoints 在本机 stability_runs 里，源码包不收录大型 checkpoint 与完整模型原始梯度；保留了小型固定输入/逐层诊断 tensor 供重新生成图表。报告和失败日志均不删改原 V2/V2.5 检查。
