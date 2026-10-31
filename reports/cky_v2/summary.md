# CKY V2：Score-first Top-R 与延迟 Gather

## 第一部分：修改前的问题与重新测量

原始源代码逐字节保存在 `v0_forest_source.py.txt`，SHA256 见 `v0_source_sha256.json`（原 `models/forest.py` 为 `0feb261ace4e55d51e95b64f6c9aeb0e6a90095d98b5c0c602f153012f6c806e`）。

原 `SemanticForest.forward` 第 72–73 行为每个 split stack 左右 semantic/coeff，第 79–81 行在 Top-R 之前展开高维配对、进行可能分配新 storage 的 flatten 并计算全配对系数和。第 91–95 行才选择结构候选；之后又 cat 全部配对向量，再 gather 少量候选。因此组合预算 12 只限制 Composer，并不限制之前的高维 staging。

在修改源码之前，已用原实现重新测量全部六种指定形状及真实缓存批次，结果为 `v0_baseline.csv`；原始 profiler 记录在 `runs/cky_v2_v0/`。其中主配置 batch 1/4 的更新中位数分别为 1311.83/1331.20 ms。没有把上一轮 `CUDA_PERFORMANCE.md` 的历史值当作本次新测量。

最终 Reference/V2 对比采用下面的严格快照协议，与初始 V0 测量轮次不同；系统波动及测量协议会影响绝对值。所有加速比只使用同轮成对测量，不跨轮次相除。

## 第二部分：V2 的实现与范围

- `models/forest_reference.py` 提取原 Cell 字典、split 枚举、分数/Mask、两级探索、Top-K、Delta 剪枝、根排序/归一化与树 backpointer，供两条实现共享。Reference 保留原高维配对展开。
- `models/forest_score_first.py` 在初筛前只保留 score 与 mask；Top-R 索引仍从原分数的 `topk` 产生，`score.gather(chosen)` 保持累计结构分数的梯度。
- 按原 metadata 的 `(offset,left_slot,right_slot)` 布局，GPU Long 查找表将 flat 索引解码成子候选 bank 地址。叶子/短区间实际槽位可能为 1、2 或 4，未机械假设每个 split 都有 K² 个配对。
- 只在选完 R 后，从现有 Cell 建立临时只读 semantic/coeff bank；每个源 Cell 在 bank 中出现一次，不展开 `split×K×K×D`。随后只 gather 所选 R 个左右向量，执行原 Composer，并按原顺序计算 `(u_L+u_R)+u_parent`。
- bank 仍需复制所有唯一源子候选，未宣称完全消除所有高维读取；删除的是全配对向量/系数物化。没有持久化 Tensorized Chart，Cell 字典及逐 split 循环保留。缓存只有 shape/device 相关的整数地址表，不缓存语义、分数或跨更新的计算图；无逐候选 `.item()`/`.cpu()`/`.tolist()`。
- 父 Span 势仍在 Top-R 之前加入。虽然它是同一 Span 的公共加数，移动顺序可能改变浮点同分行为，故没有改动。
- 原选择顺序保留：标量 Top-budget + 预算内随机探索 → 原 Composer → Top-K → beam 探索替换 → Delta → Cell 更新。原项目没有语义距离去重/多样性过滤；未删除此类阶段，也未合并同分候选。bank 的源存储去重不合并父候选身份。

默认 YAML 已启用 `model.forest_implementation: score_first`。可用 `--forest-implementation reference` 切回；同一 state_dict 可切换两条路径。旧检查点缺少该字段时按 Reference 重建；显式恢复到 V2 时只忽略这个实现选择字段，其余模型/优化配置继续严格检查。

**共同边界修复：** 原 V0 对全 `-inf` 的不可达 Span 可能保留伪合法状态并产生 NaN 先验。两条新路径均用有限势检查明确标记不可达，规范化无效叶分数字段为零；这项修复不属于加速技巧。归档 V0 在有限输入、不同 Padding 上的可观测输出/树均做了回归比较；无效分数字段先按 Mask 规范化。该非有限边界修复是相对 V0 有意扩展的行为，不能声称 V0 的 NaN 行为被保留。

Decoder、Syntax/Composer/Reader 架构、参数数量、优化器、LR 调度、数据与 CE 目标未改变。没有 V3、Triton、C++/CUDA 扩展、CUDA Graph 或 torch.compile。

## 第三部分：正确性与梯度

完整 pytest：**205 passed in 55.01s**（py312 CUDA 环境，含原有因果性、Mask、梯度、恢复及全部模式测试）。

`correctness.json` 记录 **576** 个矩阵案例：B=1/2/4，词数=4/8/16/24，K=1/2/4，R=4/12，标量/向量，两种运行模式（评估普通选择、训练探索），分别在 CPU FP32 与 CUDA BF16 上执行。另有 32 个边界案例，包含单词句、空 Mask、同分、Delta=0、全不可达/部分不可达 `-inf` 及固定 Seed 探索。

每个 Span 的 selected/retained flat 身份、顺序、分数、系数、语义，以及根、Reader、LM Logits/Loss 均比较。**这批案例所有前向字段的最大绝对误差为 0，未出现搜索结果变化。** 失败信息会指出第一个 Span、候选坐标和来源 metadata。CUDA/CPU 跨设备的 Top-K tie 顺序未被假定相同；要求是在相同设备与配置下 Reference/V2 相同。

6 组完整 LM backward 包含 tiny 标量/向量 FP32/BF16，以及真实 main_125m 的 FP32/BF16。各模块完整指标见 `gradient_comparison.json`：

| 配置 | 模式 | 精度 | 最大绝对梯度误差（所有组） | 最大归一化 L2 梯度误差（所有组） |
|---|---|---|---|---|
| tiny | forest_scalar4 | fp32 | 1.164e-10 | 9.625e-09 |
| tiny | forest_vector4 | fp32 | 9.313e-10 | 7.806e-08 |
| tiny | forest_scalar4 | bf16 | 0.000e+00 | 0.000e+00 |
| tiny | forest_vector4 | bf16 | 9.155e-05 | 3.514e-03 |
| main_125m | forest_vector4 | fp32 | 2.328e-10 | 1.267e-07 |
| main_125m | forest_vector4 | bf16 | 1.526e-05 | 3.551e-03 |

主配置 BF16 的逐模块必需指标如下；完整 FP32 与其它案例均在 JSON 中。

| 模块 | max absolute | max relative | norm Reference | norm V2 |
|---|---|---|---|---|
| syntax_encoder | 3.815e-06 | 2.000e+00 | 7.916386e-02 | 7.915979e-02 |
| span_scorer | 1.526e-05 | 1.993e+00 | 1.230233e-01 | 1.230213e-01 |
| semantic_embedding | 0.000e+00 | 0.000e+00 | 4.984792e-03 | 4.984792e-03 |
| composer | 0.000e+00 | 0.000e+00 | 2.188102e-01 | 2.188102e-01 |
| vector_coefficient_head | 1.526e-05 | 1.925e+00 | 6.133011e-02 | 6.133181e-02 |
| semantic_reader | 0.000e+00 | 0.000e+00 | 2.961155e-01 | 2.961155e-01 |
| decoder | 0.000e+00 | 0.000e+00 | 2.227521e+01 | 2.227521e+01 |

相对误差分母为 `max(|ref|,|v2|,1e-8)`，近零分量可能出现很大的相对误差；同时报告绝对误差、范数、有效量级分量的相对误差与归一化 L2 误差。BF16 向量系数 gather 的 backward 改变累积顺序，存在低精度舍入差，不是逐位一致；主配置最大归一化组 L2 差约 0.36%。FP32 主配置最大绝对梯度差约 2.33e-10。

CPU FP32 下两种模式均执行 8 个相同 AdamW Step，逐 Step 比较所有 Span 的候选身份/顺序。标量模式 Loss/参数均相同；向量模式最大 Loss 差 4.77e-7，除公共 scalar bias 外的参数最大差 3.52e-6。

**没有隐藏的逐位训练一致性承诺：** 向量模式的 `syntax.span_scorer.scalar.bias` 最终差约 1.55e-4。每棵 n 叶完全二叉树都有 2n−1 个局部势，该公共 bias 在本句根 log-softmax 中数学上完全抵消；约 1e-8 的浮点抵消残差被 AdamW 的 epsilon 放大，产生此中性方向漂移。没有冻结/清零该梯度或调整优化器。JSON 保留逐 Step bias 梯度、全部参数最大误差及非中性方向误差；该特殊中性参数允许一个峰值 auxiliary LR 的绝对差，其余参数继续使用严格容差，Loss 与搜索身份检查仍必须通过。

## 第四部分：性能

设备为 RTX 3070 Laptop 8GB、PyTorch 2.12.1+cu132、CUDA 13.2、Windows/WDDM。全部 K=4、R=12、BF16、训练模式；合成输入 H=4、T=128，真实缓存例外的实际形状显式记录。

每个案例先建立共同的模型/AdamW 快照。每次计时前恢复**完全相同的权重、Optimizer 和 Python/NumPy/Torch/CUDA RNG**，两条实现交替测量；同一行组的权重 SHA256 在 CSV 中相同。每条实现预热 2 次，普通更新同步计时 5 次。快照在 CPU，恢复、审计及 profiler 不计入计时。

普通更新时间包括完整 forward、CE、backward、clip、AdamW，输入驻留 GPU；没有数据加载/搬运、验证、Checkpoint IO 或梯度累积。不能把这些微批 Token/s 直接称为完整预训练端到端吞吐量。

CKY Forward/Backward 使用项目已有分阶段计时方法，每条路径另测 3 次；与无 profiler 的普通更新时间分开记录。以下变化按相同微批条件比较，**主配置 batch=4**：

| Metric | Reference | V2 | 改善/比率 |
|---|---|---|---|
| CKY Forward (ms) | 422.74 | 307.33 | 1.376x |
| CKY Backward (ms) | 882.37 | 693.49 | 1.272x |
| Full microbatch update (ms) | 1486.56 | 1194.92 | 1.244x |
| CE Target Tokens/s | 344.42 | 428.48 | 1.244x |
| Peak allocated GiB | 3.452 | 2.944 | 14.69% less |
| CKY Forward stack calls | 2258 | 1154 | 1.957x fewer |
| CKY Forward cat calls | 2436 | 1309 | 1.861x fewer |
| CKY Forward CUDA launches | 5704 | 4081 | 1.398x fewer |
| Full update CUDA launches | 28852 | 24241 | 1.190x fewer |
| Composer calls | 23 | 23 | unchanged |
| Composer candidate rows | 43072 | 43072 | unchanged |

全部案例（更新时间为中位数 [最小, 最大]；Forward/Backward 是单独诊断的中位数）：

| 案例 | Reference update ms | V2 update ms | update speedup | CKY forward ms | CKY backward ms | peak GiB |
|---|---|---|---|---|---|---|
| smoke_b2_w8 | 198.19 [180.61, 221.58] | 170.76 [166.97, 182.11] | 1.161x | 67.67 → 72.49 | 72.00 → 77.23 | 0.413 → 0.413 |
| smoke_b2_w16 | 638.45 [616.40, 649.68] | 519.99 [508.64, 539.83] | 1.228x | 218.33 → 177.33 | 350.36 → 283.30 | 0.517 → 0.461 |
| smoke_b2_w24 | 1373.52 [1361.12, 1409.39] | 1098.04 [1051.12, 1132.34] | 1.251x | 442.19 → 309.11 | 880.01 → 707.36 | 0.905 → 0.644 |
| smoke_b8_w24 | 1371.76 [1364.70, 1383.37] | 1082.75 [1077.95, 1102.78] | 1.267x | 438.59 → 304.12 | 877.71 → 688.37 | 2.704 → 1.686 |
| main_b1_w24 | 1453.11 [1434.15, 1515.26] | 1180.90 [1133.12, 1201.96] | 1.231x | 423.85 → 315.18 | 875.65 → 687.15 | 2.722 → 2.724 |
| main_b4_w24 | 1486.56 [1450.93, 1494.37] | 1194.92 [1174.39, 1251.95] | 1.244x | 422.74 → 307.33 | 882.37 → 693.49 | 3.452 → 2.944 |
| real_b2 | 936.40 [917.61, 955.04] | 767.30 [735.77, 775.54] | 1.220x | 310.73 → 233.65 | 570.20 → 460.73 | 0.649 → 0.512 |

真实批次使用两个不同故事、每个目标前四句历史，实际 Padding 后历史最长 20、目标最长 17，总计 29 个 CE Token。它不是 24 词全有效输入；两条实现使用同一个批次。

没有只挑最快样本。短句 8 词的分阶段时间有倒退/波动，普通更新中位数约改善 1.16x；没有宣称所有阶段普遍加速。长句完整更新改善约 1.22–1.27x。main batch=1 的峰值分配略增约 1.7 MiB（0.06%，整数地址缓存等固定开销），其它长批次显存下降更明显；没有声称每种形状都降低峰值。

### 高维物化量（main batch=4）

单位为实际分配的**元素累计量**，不是峰值字节：

| representation staging | Reference elements | V2 elements |
|---|---|---|
| before_selection_semantic | 208,625,664 | 0 |
| before_selection_coeff | 18,467,840 | 0 |
| after_selection_semantic | 200,376,320 | 66,981,888 |
| after_selection_coeff | 12,523,520 | 9,751,040 |
| total_materialized_semantic | 409,001,984 | 66,981,888 |
| total_materialized_coeff | 30,991,360 | 9,751,040 |

统计范围：Reference 初筛前的语义/系数 stack、发生 storage 改变的 expand→flatten 副本和全配对系数和；初筛后的全配对 cat、所选向量 gather。V2 统计只读源 bank 的 cat、所选左右 gather 与所选系数和。纯 view 不计，expand→flatten 是否实际复制通过 storage 地址判断；缓存的 Long 索引不是高维浮点表示。

双方共同的 embedding、父局部系数 stack/add、Composer 内部激活、最终保留/根 Padding 等不计入这项 staging 审计。元素可在不同生命周期重复分配，因此不能把累计元素乘 dtype 字节数解释为峰值显存。语义 staging 约减少 83.62%，系数约减少 68.54%。V2 初筛前这两类物化量均为零；选后仍有源 bank 成本。

算子计数来自**另一轮 profiler 下的真实完整更新**；`operator_counts.csv` 同时区分 CKY forward 和 full step。没有将 profiler 内部耗时当普通更新耗时。Composer 为每个 width 一次调用，batch=4/24 词两条路径均 23 次、43072 个候选行（包括当前实现仍计算的 Padding 行），没有减少候选预算、关闭梯度或跳过训练模块来制造加速。

## 第五部分：结论与后续边界

1. V2 保留有限可达路径的模型数学、搜索与候选身份；576 个形状/精度/模式对比的前向误差为零。低精度梯度累积及中性 bias 漂移已报告，不声称优化后的训练逐位等于 Reference。
2. 成功删除初筛前的高维笛卡尔配对物化、全配对系数和，以及选后为全部配对做的语义/系数 cat，改为只读唯一子候选 bank 加所选 R 的 gather。
3. 主配置 batch=4 的 CKY Forward 约快 1.38x、Backward 约快 1.27x、完整微批更新约快 1.24x，峰值约降 14.69%。不是至少 2x/3x 的加速承诺，也不是 Blackwell 的测量结果。
4. 剩余主要开销仍是 Python Cell/chart 调度、每个 split 的 score/mask stack、源 bank 重建和大量 autograd select/梯度累积节点。V2 只处理了高维配对这一部分，CKY backward 仍占较大耗时。
5. 建议下一项独立任务再验证 V3 的按 width Tensorized Chart，以减少剩余 scalar stack/select 与 launch；本次没有实现 V3 或任何自定义 Kernel/编译方案。

## 修改文件与复现

核心：`models/forest_reference.py`（Reference/共享搜索）、`models/forest_score_first.py`（V2）、`models/forest.py`（原接口 facade）；`models/model.py` 仅增加实现选择。

辅助：`train.py` 增加实现 CLI/兼容同权重恢复，`profile_performance.py` 增加选择 CLI，`benchmark_cky_v2.py` 与 `validate_cky_v2.py`、`tests/test_forest_v2.py`、默认 YAML 字段及文档。Decoder/Syntax/Composer/Reader 源文件均未改动。

```powershell
& 'E:\anaconda3\envs\py312\python.exe' -m pytest -q --tb=short --basetemp=.pytest_tmp_v2_full -o cache_dir=.pytest_cache_v2_full
& 'E:\anaconda3\envs\py312\python.exe' validate_cky_v2.py --output-dir reports/cky_v2
& 'E:\anaconda3\envs\py312\python.exe' benchmark_cky_v2.py --output-dir reports/cky_v2 --warmup 2 --repeats 5
& 'E:\anaconda3\envs\py312\python.exe' profile_performance.py --config configs/main_125m.yaml --batch-size 1 --lengths 24 --forest-implementation reference
& 'E:\anaconda3\envs\py312\python.exe' profile_performance.py --config configs/main_125m.yaml --batch-size 1 --lengths 24 --forest-implementation score_first
```

另已实际从旧 Reference/V0 CUDA 检查点切换到 V2 恢复训练：256 → 384 个 CE Token，累计日志为 [128,256,384]，未重启调度。见 `implementation_switch_check.json` 和 `implementation_switch_metrics.csv`。

`benchmark.csv`、`operator_counts.csv` 为汇总，`raw/*/samples.json` 和 `diagnostics.json` 保存普通/分阶段原始样本；`correctness.json`、`gradient_comparison.json` 为实际执行记录。旧 V0 逐字节源码归档供回归，不是伪代码或未实现占位。
