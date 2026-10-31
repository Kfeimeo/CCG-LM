# CKY V2.5-R：Semantic Reader FP32 隔离实验

## 1. 实验目标和范围

判断 BF16 Reader 是否放大 V2/Lazy 的微小根差异，以及 FP32 Reader 能否改善完整梯度。**本轮结果接近情况 B：Reader 内部前向误差显著缩小，但全模型梯度仅有限改善。** 不更改正式配置，保留 BF16 Reader；FP32 配置保留作实验选项。没有执行20–50步 sanity、500步训练或 V3。

main_125m、forest_vector4、B=1/4、H=4、W=24、K=4、R=12；数值对照目标长度16，性能合成目标长度128，真实 batch 沿用原数据。每个数值 case 共享同一份权重、输入与 RNG，Dropout 始终开启（model.train）。CKY、探索/Delta、Backpointer、Lazy DAG、Composer、Scorer、Decoder 结构、参数数量、优化目标与学习率均不变。数值实验直接完整 CE backward，无优化器更新。主模型实际参数为148,840,709，Reader 策略不增减参数。

## 2. 实际 Precision Policy / Dtype 验证

B1/B2：BF16 outer，FP32 IEEE Composer，Reader 随外层 BF16 AMP；B1-R/B2-R：仅 Reader 增加 FP32 IEEE 区域；E1/E2：全 FP32 IEEE 正控制。源码入口 `models/precision_policy.py::PrecisionPolicy.reader`、`configure_model`，以及 `models/reader.py::SemanticReader.forward/_forward`。FP32 Reader 在禁用 AMP 的上下文内，将 Query、候选语义、系数、log-prior 转 FP32；QKV、attention、output、norm、gate、residual 均按原公式计算。返回时 cast 到原 hidden dtype；本模型该边界实测是 FP32，后续 Decoder Linear 仍由外层 BF16 决定。没有增添可训练参数，checkpoint 对旧缺失 reader 字段按 bf16 默认兼容，显式改变策略仍拒绝 resume。

| Family | Q | K | V | logits | softmax | SDPA return | output proj | return Decoder |
|---|---|---|---|---|---|---|---|---|
| B | bfloat16 | bfloat16 | bfloat16 | float32 | float32 | bfloat16 | bfloat16 | float32 |
| R | float32 | float32 | float32 | float32 | float32 | float32 | float32 | float32 |
| E | float32 | float32 | float32 | float32 | float32 | float32 | float32 | float32 |

注意 **BF16 SDPA MATH 的 logits/softmax/内部 aggregation 本来就是 FP32**，而其 QKV 和加权结果返回为 BF16。不能把 B 描述为所有 attention 运算都 BF16。`dtype_trace.json` 保留实际 shape/dtype、bias dtype、native operator 序列和 backend policy。训练有可微 bias，沿用原 MATH 后端；`TorchDispatchMode` 只观察实际 `_safe_softmax`/bmm/dropout，未重算 attention，也未改变随机采样。单测验证观察器开启/关闭时 Reader 输出、所有输入/参数梯度及 RNG **逐位相同**。不是使用旧 `return_attention=True` 的手写诊断分支冒充 native SDPA。TF32 始终显式禁用。

## 3. Forward 数值对照

| Batch | Family | root L2 | root max abs | score max abs | prior max abs | LM logits L2 | LM max abs | CE loss abs |
|---|---|---|---|---|---|---|---|---|
| 1 | B | 2.95e-07 | 1.19e-06 | 0 | 0 | 0.00614 | 0.0156 | 0.0004 |
| 1 | R | 2.95e-07 | 1.19e-06 | 0 | 0 | 0.00347 | 0.0156 | 0.00107 |
| 1 | E | 2.82e-07 | 1.1e-06 | 0 | 0 | 4.75e-07 | 1.25e-06 | 9.54e-07 |
| 4 | B | 3.37e-07 | 1.49e-06 | 0 | 0 | 0.00537 | 0.0176 | 0.000343 |
| 4 | R | 3.37e-07 | 1.49e-06 | 0 | 0 | 0.00318 | 0.0156 | 1.14e-05 |
| 4 | E | 3.3e-07 | 1.67e-06 | 0 | 0 | 6.26e-07 | 2.15e-06 | 0 |

B 与 R 的每个实现，其 root semantic、vector coefficient、log-prior **实际逐位不变**；候选与回溯完全一致。Reader 精度没有反馈到树或搜索。LM logits 相对 L2 在 batch1/4 均下降，但 CE 的单次绝对差并非单调改善：batch1 的 R loss 差反而大于 B；不把某个指标的降低外推为整体等价。原始全部 span selected-flat、retained Candidate ID、左右子节点 ID、根 ID 与 RNG 重放明细保存在 `search_b*_*.json` / `rng_b*.pt`（grad-enabled forward replay，只归档离散结构，不作为替换输入的 backward 证据）。

## 4. Reader 内部误差传播

首个 Reader（layer4）下表显示相同精度下的 V2/Lazy 配对误差；后续 Reader 的所有相同字段、shape、dtype 见 `case_b*.json`。

| Batch | Position | B max abs | R max abs | B relative L2 | R relative L2 | R RMS |
|---|---|---|---|---|---|---|
| 1 | query_input | 0 | 0 | 0 | 0 | 0 |
| 1 | query_projection | 0 | 0 | 0 | 0 | 0 |
| 1 | key_input | 1.19e-06 | 1.19e-06 | 8.47e-08 | 8.47e-08 | 2.78e-07 |
| 1 | key_projection | 0.000488 | 1.43e-06 | 2.36e-06 | 1.34e-07 | 2.51e-07 |
| 1 | value_input | 1.19e-06 | 1.19e-06 | 2.95e-07 | 2.95e-07 | 2.95e-07 |
| 1 | value_projection | 0.00391 | 7.75e-07 | 0.000108 | 3.31e-07 | 1.88e-07 |
| 1 | attention_logits | 3.46e-05 | 9.54e-07 | 6.97e-07 | 9.8e-08 | 2.1e-07 |
| 1 | attention_probabilities | 3.65e-06 | 8.94e-08 | 1.96e-06 | 2.04e-07 | 1.47e-08 |
| 1 | attention_aggregation_internal | 0.000623 | 4.17e-07 | 5.58e-05 | 2.78e-07 | 8.99e-08 |
| 1 | attention_weighted_values | 0.00195 | 4.17e-07 | 0.000117 | 2.78e-07 | 8.99e-08 |
| 1 | output_projection | 0.00195 | 2.68e-07 | 0.000629 | 3.31e-07 | 6.24e-08 |
| 1 | residual_output | 0.000244 | 2.38e-07 | 1.56e-05 | 2.04e-08 | 1.87e-08 |
| 1 | return_to_decoder | 0.000244 | 2.38e-07 | 1.56e-05 | 2.04e-08 | 1.87e-08 |
| 4 | query_input | 0 | 0 | 0 | 0 | 0 |
| 4 | query_projection | 0 | 0 | 0 | 0 | 0 |
| 4 | key_input | 1.49e-06 | 1.49e-06 | 1.03e-07 | 1.03e-07 | 3.17e-07 |
| 4 | key_projection | 0.0156 | 9.54e-07 | 4.8e-05 | 1.35e-07 | 2.39e-07 |
| 4 | value_input | 1.49e-06 | 1.49e-06 | 3.37e-07 | 3.37e-07 | 3.37e-07 |
| 4 | value_projection | 0.00781 | 1.19e-06 | 0.000122 | 3.95e-07 | 2.28e-07 |
| 4 | attention_logits | 0.00183 | 9.54e-07 | 1.52e-05 | 9.6e-08 | 2.03e-07 |
| 4 | attention_probabilities | 0.000118 | 1.19e-07 | 2.82e-05 | 1.96e-07 | 1.41e-08 |
| 4 | attention_aggregation_internal | 0.000667 | 5.96e-07 | 7.06e-05 | 3.33e-07 | 9.67e-08 |
| 4 | attention_weighted_values | 0.00391 | 5.96e-07 | 0.000211 | 3.33e-07 | 9.67e-08 |
| 4 | output_projection | 0.00391 | 2.98e-07 | 0.000732 | 4.01e-07 | 6.67e-08 |
| 4 | residual_output | 0.000488 | 2.38e-07 | 1.63e-05 | 2.2e-08 | 1.98e-08 |
| 4 | return_to_decoder | 0.000488 | 2.38e-07 | 1.63e-05 | 2.2e-08 | 1.98e-08 |

![Reader propagation](figures/reader_error_propagation.png)

图中精确零值只为对数显示置于1e-12，原始 JSON 保留0。带 mask 的 attention logits 中匹配的 -inf 被标注为预期 masked infinities；范数/RMS 只在双方有限位置计算，任何不匹配非有限值单独计数，未把 masked -inf 作为模型溢出。

**实际观察：** FP32 Reader 显著减小 K/V 投影、attention 和 Reader residual 的配对差异。首个 Reader 的 Query 原本完全相同；batch1 的 Value Projection max abs 从3.90625e-3降到7.7486e-7，而 batch4 的 Key max abs 比 Value 更大。因此敏感点首先出现在低精度 K/V 投影及返回边界，不能把 Value 永久指定为所有输入的唯一最大误差源。

Reader 后第一个 Decoder Block（表中为1-based层号）：

| Batch | Family | Decoder block | max abs | relative L2 | RMS | dtype pair | shape |
|---|---|---|---|---|---|---|---|
| 1 | B | 5 | 0.00977 | 0.00123 | 0.00127 | ['torch.float32', 'torch.float32'] | [1, 16, 768] |
| 1 | B | 9 | 0.0246 | 0.00366 | 0.0051 | ['torch.float32', 'torch.float32'] | [1, 16, 768] |
| 1 | B | 13 | 0.0399 | 0.0047 | 0.00787 | ['torch.float32', 'torch.float32'] | [1, 16, 768] |
| 1 | R | 5 | 2.38e-07 | 2.12e-08 | 2.18e-08 | ['torch.float32', 'torch.float32'] | [1, 16, 768] |
| 1 | R | 9 | 4.77e-07 | 2.66e-08 | 3.7e-08 | ['torch.float32', 'torch.float32'] | [1, 16, 768] |
| 1 | R | 13 | 0.0178 | 0.00186 | 0.00311 | ['torch.float32', 'torch.float32'] | [1, 16, 768] |
| 1 | E | 5 | 7.15e-07 | 1.23e-07 | 1.26e-07 | ['torch.float32', 'torch.float32'] | [1, 16, 768] |
| 1 | E | 9 | 1.67e-06 | 2.72e-07 | 3.82e-07 | ['torch.float32', 'torch.float32'] | [1, 16, 768] |
| 1 | E | 13 | 2.38e-06 | 3.41e-07 | 5.8e-07 | ['torch.float32', 'torch.float32'] | [1, 16, 768] |
| 4 | B | 5 | 0.0156 | 0.00132 | 0.00135 | ['torch.float32', 'torch.float32'] | [4, 16, 768] |
| 4 | B | 9 | 0.0259 | 0.00329 | 0.00454 | ['torch.float32', 'torch.float32'] | [4, 16, 768] |
| 4 | B | 13 | 0.0378 | 0.0041 | 0.00684 | ['torch.float32', 'torch.float32'] | [4, 16, 768] |
| 4 | R | 5 | 2.38e-07 | 2.24e-08 | 2.29e-08 | ['torch.float32', 'torch.float32'] | [4, 16, 768] |
| 4 | R | 9 | 0.0186 | 0.00123 | 0.0017 | ['torch.float32', 'torch.float32'] | [4, 16, 768] |
| 4 | R | 13 | 0.0297 | 0.002 | 0.00334 | ['torch.float32', 'torch.float32'] | [4, 16, 768] |
| 4 | E | 5 | 8.34e-07 | 1.5e-07 | 1.53e-07 | ['torch.float32', 'torch.float32'] | [4, 16, 768] |
| 4 | E | 9 | 2.38e-06 | 3.45e-07 | 4.76e-07 | ['torch.float32', 'torch.float32'] | [4, 16, 768] |
| 4 | E | 13 | 3.34e-06 | 4.41e-07 | 7.35e-07 | ['torch.float32', 'torch.float32'] | [4, 16, 768] |

R 的早期 block 仍可保持很小误差，较深层/最终 logits 则重新出现明显差异；并非每个紧随 Reader 的 block 都立即大幅放大。全 FP32 正控制明显更一致。**推断：** BF16 下游的量化/反向路径仍是敏感区域；本轮没有证明具体 cuBLAS 算法或某个 Decoder 子算子是唯一根因。

相同 Reader 输入控制：只在 `torch.no_grad()` 中把 semantic、coefficient、log-prior、mask 替换为同一预缓存参考。Dropout 保持训练态，Query/Decoder 输入未人工替换。此时输出和后续 Query 的对照如下：

| Batch | Family | shared memory | LM max abs | all entry RNG equal | max query input abs |
|---|---|---|---|---|---|
| 1 | B | False | 0.0156 | True | 0.04058837890625 |
| 1 | B | True | 0 | True | 0.0 |
| 1 | R | False | 0.0156 | True | 0.0118408203125 |
| 1 | R | True | 0 | True | 0.0 |
| 1 | E | False | 1.43e-06 | True | 2.86102294921875e-06 |
| 1 | E | True | 0 | True | 0.0 |
| 4 | B | False | 0.0176 | True | 0.04583740234375 |
| 4 | B | True | 0 | True | 0.0 |
| 4 | R | False | 0.0156 | True | 0.03215479850769043 |
| 4 | R | True | 0 | True | 0.0 |
| 4 | E | False | 2.15e-06 | True | 4.291534423828125e-06 |
| 4 | E | True | 0 | True | 0.0 |

缓存输入时，Reader 输出与最终 logits 实测完全一致，所有 Reader 入口 RNG 一致。它支持“正常差异由不完全相同的根输入触发”的解释，但 **不用于宣称完整模型 backward 等价**。该无梯度控制沿用原 automatic SDPA selector，未假设其后端与有梯度 MATH 相同；控制和正常 backward 数据分开保存。

## 5. 完整 Backward 梯度对照

| Batch | Family | grad norm V2 | grad norm Lazy | max abs | global relative L2 | strict failed params | nonfinite grads |
|---|---|---|---|---|---|---|---|
| 1 | B | 22.3 | 22.3 | 0.00466 | 0.00868 | 27 | 0 |
| 1 | R | 22.3 | 22.3 | 0.00434 | 0.00787 | 24 | 0 |
| 1 | E | 21.7 | 21.7 | 3.58e-07 | 6.22e-07 | 0 | 0 |
| 4 | B | 12.8 | 12.8 | 0.00232 | 0.00704 | 7 | 0 |
| 4 | R | 12.8 | 12.8 | 0.00235 | 0.00657 | 6 | 0 |
| 4 | E | 12.4 | 12.4 | 2.22e-07 | 7.39e-07 | 0 | 0 |

全模型梯度改善比 B/R：B=1 为 **1.104x**，B=4 为 **1.071x**。对应相对 L2 降低约 9.39% / 6.67%，没有声称消除完整梯度差异。比值只在有效非零分母时计算，同时保留绝对误差；本轮 R 分母并不接近零。

| Batch | Module | B grad L2 | R grad L2 | B max abs | R max abs | R significant relative |
|---|---|---|---|---|---|---|
| 1 | syntax_encoder | 0.0107 | 0.00872 | 8.58e-06 | 7.63e-06 | 0.365 |
| 1 | span_scorer | 0.0116 | 0.00917 | 7.63e-05 | 4.6e-05 | 1.87 |
| 1 | semantic_embedding | 0.00654 | 0.00514 | 2.73e-06 | 2.85e-06 | 0.103 |
| 1 | composer | 0.00637 | 0.00492 | 2.06e-05 | 1.81e-05 | 0.899 |
| 1 | vector_coefficient_head | 0.0122 | 0.00891 | 7.63e-05 | 4.6e-05 | 1.87 |
| 1 | semantic_reader | 0.00679 | 0.00466 | 2.24e-05 | 1.9e-05 | 1.02 |
| 1 | decoder | 0.00868 | 0.00787 | 0.00466 | 0.00434 | 2 |
| 4 | syntax_encoder | 0.00763 | 0.00604 | 5.01e-06 | 3.81e-06 | 0.202 |
| 4 | span_scorer | 0.00692 | 0.00521 | 6.1e-05 | 6.1e-05 | 0.692 |
| 4 | semantic_embedding | 0.00563 | 0.00389 | 6.05e-07 | 5.02e-07 | 0.0382 |
| 4 | composer | 0.00562 | 0.00383 | 7.49e-06 | 5.79e-06 | 0.325 |
| 4 | vector_coefficient_head | 0.00977 | 0.00783 | 1.81e-05 | 1.72e-05 | 0.692 |
| 4 | semantic_reader | 0.00589 | 0.00378 | 2.56e-05 | 4.16e-06 | 0.268 |
| 4 | decoder | 0.00704 | 0.00657 | 0.00232 | 0.00235 | 2 |

`gradient_modules.csv` 对七个模块、三种精度、两种 batch 保留两侧完整梯度范数与全部五项误差指标（42行）。上表用于直接比较 B/R；例如 Reader/Composer/Syntax 均改善，但 Decoder 的主导梯度只小幅改善。vector head 是 span scorer 的子集，模块表有重叠，全局统计不重复计数。`gradient_comparison.json` 还保留逐参数两侧范数变化、原始最大相对误差、显著梯度误差（|grad|≥1e-5）、严格失败参数名与非有限梯度数。原 V2 容差不变：BF16 atol=.00015/rtol=.04；全 FP32 atol=3e-6/rtol=.0003。近零梯度不只靠逐元素 relative 判断。全局统计直接对全部独立参数计算，未重复计算 vector head 子集。**严格 BF16 梯度不匹配仍保留，不以放宽标准将其转成通过。**

![Full gradients](figures/full_gradient_error.png)

## 6. Benchmark 与显存成本

同设备、同轮次四臂 B1/B2/B1-R/B2-R，所有参数/AdamW/RNG 快照相同，均为 BF16 outer、FP32 IEEE Composer。每臂 warmup2、有效样本8，四臂旋转/反转顺序平衡位置。输入驻 GPU，正常 forward + CE backward + clip + AdamW，正确 CUDA 同步；setup/恢复、dtype audit、profiler 不计普通时间。沿用共同零梯度 AdamW 初始化 buffers 的快照协议，初始化中的 weight decay 对四臂相同。实际环境在 `benchmark_environment.json`，未与旧轮次数据直接计算任何比值。

| Case | Arm | update median [min,max] ms | CE/s | peak GiB | Reader F/B ms | Forest F/B ms | full launches |
|---|---|---|---|---|---|---|---|
| main_b1_w24 | B1 | 1097.1 [1079.6,1139.3] | 116.7 | 2.7257 | 17.56/17.13 | 302.2/641.0 | 24013 |
| main_b1_w24 | B2 | 775.6 [767.4,810.3] | 165.0 | 2.7257 | 19.58/21.64 | 320.0/429.5 | 14852 |
| main_b1_w24 | B1-R | 1093.9 [1078.4,1156.7] | 117.0 | 2.7258 | 17.71/18.61 | 346.4/788.5 | 23901 |
| main_b1_w24 | B2-R | 778.5 [755.3,803.7] | 164.4 | 2.7265 | 17.34/18.52 | 324.3/410.3 | 14740 |
| main_b4_w24 | B1 | 1123.4 [1094.1,1173.4] | 455.8 | 3.1023 | 17.95/16.97 | 293.5/649.9 | 23961 |
| main_b4_w24 | B2 | 805.4 [790.6,833.1] | 635.7 | 2.7378 | 23.95/23.40 | 335.6/442.4 | 14908 |
| main_b4_w24 | B1-R | 1134.4 [1092.0,1161.8] | 451.3 | 3.1011 | 17.92/17.66 | 376.0/819.4 | 23861 |
| main_b4_w24 | B2-R | 804.0 [762.5,824.1] | 636.8 | 2.7348 | 17.64/17.53 | 340.2/436.4 | 14808 |
| real_b2 | B1 | 718.6 [685.4,779.7] | 40.4 | 0.5640 | 8.72/10.88 | 241.8/435.0 | 15295 |
| real_b2 | B2 | 493.0 [483.1,510.4] | 58.8 | 0.4081 | 9.93/10.72 | 247.2/278.2 | 9683 |
| real_b2 | B1-R | 702.2 [695.4,738.5] | 41.3 | 0.5640 | 9.53/9.55 | 279.8/513.5 | 15257 |
| real_b2 | B2-R | 502.3 [479.5,545.6] | 57.7 | 0.4081 | 8.87/9.38 | 237.1/271.3 | 9645 |
| smoke_b2_w16 | B1 | 507.6 [492.9,923.4] | 504.4 | 0.4962 | 8.69/9.74 | 167.4/271.2 | 9663 |
| smoke_b2_w16 | B2 | 363.8 [353.8,515.4] | 703.7 | 0.4142 | 11.24/10.54 | 175.3/184.7 | 6681 |
| smoke_b2_w16 | B1-R | 508.3 [481.6,2248.7] | 503.7 | 0.4963 | 8.18/9.10 | 183.4/323.0 | 9629 |
| smoke_b2_w16 | B2-R | 391.0 [350.0,434.2] | 654.8 | 0.4142 | 8.90/9.92 | 183.7/183.0 | 6647 |
| smoke_b2_w24 | B1 | 1010.8 [988.3,1069.1] | 253.3 | 0.7274 | 9.02/9.40 | 298.6/629.2 | 22983 |
| smoke_b2_w24 | B2 | 673.8 [651.9,731.4] | 379.9 | 0.4161 | 10.61/11.66 | 327.4/410.0 | 13885 |
| smoke_b2_w24 | B1-R | 1004.5 [986.6,1032.2] | 254.9 | 0.7275 | 8.80/9.75 | 337.3/795.9 | 22949 |
| smoke_b2_w24 | B2-R | 663.4 [651.6,725.5] | 385.9 | 0.4162 | 8.70/8.60 | 317.5/413.6 | 13851 |

同一个 repeat 的配对时间比，而非跨测量轮次：

| Case | B V2/Lazy | R V2/Lazy | Lazy R/B extra cost | R/B sample range |
|---|---|---|---|---|
| main_b1_w24 | 1.423x | 1.426x | -0.91% | [-2.69%,+4.73%] |
| main_b4_w24 | 1.395x | 1.394x | -0.87% | [-4.84%,+3.09%] |
| real_b2 | 1.451x | 1.407x | +0.93% | [-4.32%,+10.43%] |
| smoke_b2_w16 | 1.381x | 1.345x | +5.05% | [-32.10%,+22.73%] |
| smoke_b2_w24 | 1.487x | 1.524x | -1.98% | [-9.85%,+9.47%] |

![Full update](figures/full_update.png)

小幅负额外成本属于本轮采样波动，不能据此认定 FP32 Reader 更快。smoke W16 保留了最高2248.7ms的 B1-R 离群样本，未事后删除，其配对成本区间也很宽。其余 shape 的低额外成本只适用于本机这些 batch。Lazy 主模型 peak 基本相当，几 MiB 内差异不解释为稳定显存收益。

独立 profiler 补充重放（每臂单次，先 warmup2；相同参数快照 hash，**不用于任何加速比**）：

| Case | Arm | profiled update wall ms | CUDA envelope ms |
|---|---|---|---|
| main_b1_w24 | B1 | 1669.7 | 1669.0 |
| main_b1_w24 | B2 | 1023.7 | 1022.9 |
| main_b1_w24 | B1-R | 1551.0 | 1550.1 |
| main_b1_w24 | B2-R | 1140.8 | 1139.9 |
| main_b4_w24 | B1 | 1852.3 | 1848.3 |
| main_b4_w24 | B2 | 1171.1 | 1170.2 |
| main_b4_w24 | B1-R | 1710.3 | 1708.6 |
| main_b4_w24 | B2-R | 1012.3 | 1011.4 |
| real_b2 | B1 | 1004.7 | 1003.9 |
| real_b2 | B2 | 682.3 | 681.5 |
| real_b2 | B1-R | 1159.6 | 1158.8 |
| real_b2 | B2-R | 785.4 | 784.6 |
| smoke_b2_w16 | B1 | 877.9 | 867.4 |
| smoke_b2_w16 | B2 | 561.3 | 560.5 |
| smoke_b2_w16 | B1-R | 663.5 | 662.8 |
| smoke_b2_w16 | B2-R | 485.4 | 484.6 |
| smoke_b2_w24 | B1 | 1476.7 | 1475.7 |
| smoke_b2_w24 | B2 | 895.9 | 894.9 |
| smoke_b2_w24 | B1-R | 1388.0 | 1387.1 |
| smoke_b2_w24 | B2-R | 953.9 | 953.2 |

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

回归检查：`288 passed, 1 skipped in 58.57s`。数值诊断和性能脚本独立，避免在 benchmark 时并发 GPU 任务。与上一轮的差异位置是观察事实；“BF16 下游放大”和“更小精度边界可能有帮助”仅是相应证据下的解释/未测假说。
