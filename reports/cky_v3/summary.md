# CKY V3：Width-batched GPU Score-only Tensorization

## 1. 原有瓶颈

实现可切换的 `lazy_tensorized`，保留所有旧实现和默认 `score_first`。有界beam结构、前向及原严格梯度通过，并测得完整更新收益。

实施前审查见 `code_review.md` / `feasibility.json`。旧Lazy的Top-R/Top-K/探索本来已按width批量执行；主要残留是 `_search_scores` 的offset循环、四类score/mask stack、pair cat、逐Span Cell/view与对应反向节点。每width有n-w+1个父Span，没有把旧实现误称为逐Span Top-K。

实施前新测main B4 Lazy=614.11ms（warmup2/8样本），保存在prechange_baseline，仅用于定位。后文倍率均使用各自同轮配对，不拿此值或旧报告226ms直接相除。

## 2. Tensorized Chart Layout

每width保存全新可微 `[B*H,N_w,k_w]` score与动态bool mask。k1=1，Ew是各split实际左右槽数乘积之和，proposal=min(Ew,R)，kw=min(proposal,K)。无效性不缩短物理槽位。GPU几何表直接对应offset/left-slot/right-slot的legacy ragged顺序，不使用固定K²编号。

Candidate ID仍为span base+row*实际槽数+slot，base按width/start排列，包含无效槽。B缓存左右ID解码与root gather；正常forward不创建逐Span Cell，仅return_spans诊断展开字典。直接提供原layouts/bands，不重新物化大量Cell适配DAG。

缓存key含layout version、W、B*H、K/R、device/index dtype、decode模式；LRU最多8项，device/dtype迁移清空。不缓存score、神经输出、动态图mask或跨step语义图，不增加参数/state_dict字段。

## 3. GPU Candidate Scoring

每width通过packed score/mask gather处理全部 `[B*H,N_w,Ew]`，保持 `(left+right).float()+local`。width2先做原BF16叶子相加再float；更长width至少含一个FP32内部子树，bank提升与原运算一致。未强制全部score FP32，未改变加法顺序。每层新建Tensor，不原地覆盖需要backward的值。

| Case | Precision | audit hit | cumulative hit rate | cold build host ms | cache MiB |
|---|---|---|---|---|---|
| main_b1_w24 | bf16 | True | 0.929 | 33.10 | 2.361 |
| main_b4_w24 | bf16 | True | 0.929 | 41.42 | 6.440 |
| real_b2 | bf16 | True | 0.929 | 45.68 | 1.937 |
| smoke_b2_w16 | bf16 | True | 0.929 | 12.33 | 0.846 |
| smoke_b2_w24 | bf16 | True | 0.929 | 34.51 | 3.721 |
| smoke_b2_w8 | bf16 | True | 0.929 | 3.65 | 0.050 |
| smoke_b8_w24 | bf16 | True | 0.929 | 45.84 | 11.878 |
| main_b1_w24 | fp32 | True | 0.929 | 34.24 | 2.361 |
| main_b4_w24 | fp32 | True | 0.929 | 37.50 | 6.440 |

正式计时已预热，固定shape缓存命中；累计hit rate包含首次miss。build是主机规划/H2D提交时间，非纯GPU active time。

## 4. Candidate Selection

原生topk保留同compact形状、顺序及末维stride，无新ID排序。执行顺序仍为Top-R预算拆分、proposal探索、可微score gather、Top-K、末槽探索替换、Delta。三个rand_like按原width顺序和相同形状/dtype执行；Python/NumPy/CPU/CUDA RNG逐项核对，不依赖旧探索重放。

B保留已经batched的选择，只优化ID解码、Cell/root装配。完全同分/近同分、欠满K/R、无效槽及Delta=0均测试。唯一兼容分支为额外的K=None且R=None精确Catalan枚举：近零score梯度累计曾差7.629e-6，超过严格3e-6/3e-4。保留失败测试和logs/v3_b_unbounded_strict_failure.log，使用原Lazy搜索保持其梯度，不放宽标准。正式K=1/2/4、R=4/12仍走张量化。

## 5. Autograd 正确性

选中score和root log-prior保留到SpanScorer的可微引用；仅原离散priority detach。独立prior backward验证Scorer非零。共享DAG、空闭包的原Composer零梯度anchor保持，未冻结中性bias。

`_finish_lazy` 提取旧尾部；拓扑D2H、CPU闭包/memoization、plan上传、width Composer、根语义/系数组装主体与冻结源码逐字节相同，见source_scope_audit.json。冻结原实现和提取后的输出/输入与参数梯度逐位回归通过。神经模块源码/参数名称/数量保持；没有no-grad搜索或score reconstruction。

## 6. Forward / Backward 验证

1080矩阵+36边界：B=1/2/4，W=1/4/8/16/24，K=1/2/4，R=4/12，Scalar/Vector，CPU FP32、CUDA IEEE FP32、BF16 outer，Eval/Train探索。逐Span比较metadata、selected/retained flat、左右ID、base、score/mask及树次序；根score/prior/coeff/semantic和完整LM前向逐位相同。BF16测试score/coeff实际保存为BF16。无历史/无效history另有模型单测，原score_first/Lazy交叉检查也保留。

| Device | Precision | Model/mode | Batch | global grad L2 | max abs | strict failed params | nonfinite |
|---|---|---|---|---|---|---|---|
| cpu | fp32 | forest_scalar4 | 2 | 3.4e-09 | 3.73e-09 | 0 | 0 |
| cpu | fp32 | forest_vector4 | 2 | 1.7e-09 | 1.86e-09 | 0 | 0 |
| cuda | fp32 | forest_scalar4 | 2 | 3.9e-09 | 1.86e-09 | 0 | 0 |
| cuda | fp32 | forest_vector4 | 2 | 3.46e-09 | 1.86e-09 | 0 | 0 |
| cuda | bf16 | forest_scalar4 | 2 | 4.81e-05 | 6.1e-05 | 0 | 0 |
| cuda | bf16 | forest_vector4 | 2 | 0.000145 | 0.000122 | 0 | 0 |
| cuda | fp32 | main | 1 | 3.69e-10 | 3.49e-10 | 0 | 0 |
| cuda | fp32 | main | 4 | 8.22e-10 | 4.66e-10 | 0 | 0 |
| cuda | bf16 | main | 1 | 3.34e-12 | 2.91e-11 | 0 | 0 |
| cuda | bf16 | main | 4 | 1.73e-05 | 7.63e-06 | 0 | 0 |

旧容差不变：FP32 atol=3e-6/rtol=.0003，BF16 atol=.00015/rtol=.04。10组完整LM零严格失败，此结论针对旧Lazy与V3，不覆盖上一轮V2/Lazy已有BF16差异。前向max abs/L2/RMS在JSON，七模块范数和显著梯度误差在CSV；vector head是span scorer子集，全模型不重复计数。

| Precision | Batch | Module | grad norm Lazy | grad norm V3 | max abs | relative L2 | significant relative |
|---|---|---|---|---|---|---|---|
| fp32 | 1 | syntax_encoder | 0.0622 | 0.0622 | 1.16e-10 | 1e-07 | 4.35e-06 |
| fp32 | 1 | span_scorer | 0.0972 | 0.0972 | 3.49e-10 | 5.18e-08 | 2.88e-05 |
| fp32 | 1 | semantic_embedding | 0.00419 | 0.00419 | 0 | 0 | 0 |
| fp32 | 1 | composer | 0.18 | 0.18 | 0 | 0 | 0 |
| fp32 | 1 | vector_coefficient_head | 0.0481 | 0.0481 | 0 | 0 | 0 |
| fp32 | 1 | semantic_reader | 0.238 | 0.238 | 0 | 0 | 0 |
| fp32 | 1 | decoder | 21.7 | 21.7 | 0 | 0 | 0 |
| fp32 | 4 | syntax_encoder | 0.0418 | 0.0418 | 9.46e-11 | 1.85e-07 | 6.11e-06 |
| fp32 | 4 | span_scorer | 0.0753 | 0.0753 | 4.66e-10 | 7.46e-08 | 8.49e-06 |
| fp32 | 4 | semantic_embedding | 0.00248 | 0.00248 | 7.28e-12 | 4.79e-08 | 4.53e-07 |
| fp32 | 4 | composer | 0.107 | 0.107 | 1.16e-10 | 3.38e-08 | 5.46e-06 |
| fp32 | 4 | vector_coefficient_head | 0.0333 | 0.0333 | 1.16e-10 | 2.51e-08 | 4.68e-06 |
| fp32 | 4 | semantic_reader | 0.151 | 0.151 | 0 | 0 | 0 |
| fp32 | 4 | decoder | 12.4 | 12.4 | 0 | 0 | 0 |
| bf16 | 1 | syntax_encoder | 0.0792 | 0.0792 | 0 | 0 | 0 |
| bf16 | 1 | span_scorer | 0.123 | 0.123 | 2.91e-11 | 6.04e-10 | 3.33e-07 |
| bf16 | 1 | semantic_embedding | 0.00498 | 0.00498 | 0 | 0 | 0 |
| bf16 | 1 | composer | 0.219 | 0.219 | 0 | 0 | 0 |
| bf16 | 1 | vector_coefficient_head | 0.0614 | 0.0614 | 0 | 0 | 0 |
| bf16 | 1 | semantic_reader | 0.296 | 0.296 | 0 | 0 | 0 |
| bf16 | 1 | decoder | 22.3 | 22.3 | 0 | 0 | 0 |
| bf16 | 4 | syntax_encoder | 0.0644 | 0.0644 | 1.91e-06 | 0.00219 | 0.0761 |
| bf16 | 4 | span_scorer | 0.134 | 0.134 | 7.63e-06 | 0.00128 | 0.187 |
| bf16 | 4 | semantic_embedding | 0.00244 | 0.00244 | 1.09e-11 | 5.62e-08 | 6.57e-07 |
| bf16 | 4 | composer | 0.105 | 0.105 | 1.16e-10 | 3.34e-08 | 2.88e-06 |
| bf16 | 4 | vector_coefficient_head | 0.0336 | 0.0336 | 7.63e-06 | 0.00145 | 0.129 |
| bf16 | 4 | semantic_reader | 0.146 | 0.146 | 0 | 0 | 0 |
| bf16 | 4 | decoder | 12.8 | 12.8 | 0 | 0 | 0 |

| Experiment | Steps | max loss difference | max parameter difference | max neutral-bias difference |
|---|---|---|---|---|
| cpu_forest_scalar4.json | 8 | 4.77e-07 | 1.12e-05 | 1.12e-05 |
| cpu_forest_vector4.json | 8 | 0 | 4.55e-06 | 4.55e-06 |
| cuda_real_training.json | 20 | 0.00144 | 0.00196 | 4.43e-05 |

CPU FP32每模式8步同batch/RNG；真实CUDA从旧B2 seed42/500步checkpoint续训至520步，8000到8320 CE tokens，优化器/LR/数据指纹连续。GPU重放验证两臂统一确定性算法，与普通benchmark设置区分。V3第510步保存后第511步重放，参数、Optimizer矩、RNG、scheduler和sampler逐位一致。短训练仍有权重漂移，不冻结中性bias，不以20步证明预训练质量等价。

轨迹中的search/probe_search沿用旧helper的合计比较，包含数值score；不能由非零changed_entries直接认定Candidate ID或树结构改变。固定权重的身份一致性在矩阵中独立验证，训练过程没有声称逐步参数或搜索统计完全相同。

真实默认microbatch4的20步均只用1个microbatch，Lazy/V3为54.73/78.05 CE/s。另以相同effective CE=16、显式microbatch1独立跑20步，实际每步micro_batches=[2, 3]，Lazy/V3为26.88/33.81 CE/s。后者确实含mmap取样、Collator、H2D、多microbatch梯度累积、clip和AdamW；排除保存和独立probe，不与驻GPU微批吞吐混称。microbatch切换沿用恢复接口警告，两臂共享分块/输入/RNG，effective预算、LR和优化器未变。详见training/与accumulation_training/。

![Training](figures/training.png)

回归：`413 passed, 1 skipped in 76.41s (0:01:16)`。初始诊断Hook布尔参数异常及无界严格梯度失败日志保留。

## 7. GPU 性能

RTX3070 Laptop8GB，torch2.12.1+cu132，CUDA13.2。七种BF16形状+两种全FP32主模型，每侧warmup2、有效8样本，交替顺序、恢复同一物理Model/AdamW/scaler/RNG；共同零梯度AdamW buffers setup含相同weight decay，setup/恢复不计时。输入驻GPU，计时含CE forward/backward/clip/AdamW。真实batch H4/W20/T17、CE29，数据未变。完整min/max、组件分布、输入/权重hash均保留。

| Case | Precision | Lazy median [min,max] ms | V3 median [min,max] ms | paired ratio median | CE/s Lazy / V3 |
|---|---|---|---|---|---|
| main_b1_w24 | bf16 | 522.3 [501.7,558.2] | 283.7 [277.0,291.0] | 1.836x | 245.1 / 451.2 |
| main_b4_w24 | bf16 | 545.6 [517.5,593.5] | 297.1 [289.6,311.4] | 1.831x | 938.3 / 1723.2 |
| real_b2 | bf16 | 335.0 [324.9,373.7] | 171.2 [167.8,174.2] | 1.964x | 86.6 / 169.4 |
| smoke_b2_w16 | bf16 | 258.3 [237.0,296.3] | 152.3 [145.3,190.4] | 1.705x | 991.2 / 1680.7 |
| smoke_b2_w24 | bf16 | 464.9 [442.8,478.4] | 206.8 [200.9,222.7] | 2.210x | 550.6 / 1237.6 |
| smoke_b2_w8 | bf16 | 113.5 [109.8,126.7] | 91.0 [87.5,104.1] | 1.224x | 2255.6 / 2812.3 |
| smoke_b8_w24 | bf16 | 480.4 [462.7,499.9] | 223.3 [213.2,239.3] | 2.103x | 2131.7 / 4586.7 |
| main_b1_w24 | fp32 | 508.6 [492.3,552.1] | 274.3 [253.6,311.0] | 1.837x | 251.7 / 466.6 |
| main_b4_w24 | fp32 | 520.7 [506.4,548.6] | 278.5 [265.8,305.9] | 1.914x | 983.3 / 1838.1 |

![Full update](figures/full_update.png)

以下独立组件各3次，单位ms，不可相加还原整步。Score-only backward使用完整LM真实log-prior边界梯度，两臂相同；独立Span势作为叶子仅用于隔离诊断，正常完整更新没有detach。

| Case | Precision | Score F Lazy/V3 ms | Score B | Forest F | Forest B | DAG adapter host |
|---|---|---|---|---|---|---|
| main_b1_w24 | bf16 | 152.68 / 72.90 | 181.42 / 28.43 | 188.38 / 114.23 | 220.33 / 91.74 | 2.01 / 2.07 |
| main_b4_w24 | bf16 | 149.10 / 72.84 | 211.72 / 34.11 | 192.51 / 154.30 | 222.30 / 94.88 | 5.75 / 5.37 |
| real_b2 | bf16 | 118.19 / 62.41 | 111.93 / 21.77 | 144.46 / 91.45 | 147.99 / 64.64 | 2.81 / 2.00 |
| smoke_b2_w16 | bf16 | 83.70 / 53.21 | 68.45 / 19.34 | 101.03 / 81.73 | 97.52 / 55.79 | 2.30 / 2.26 |
| smoke_b2_w24 | bf16 | 166.90 / 74.19 | 187.86 / 27.86 | 207.58 / 122.46 | 232.71 / 90.19 | 3.01 / 2.95 |
| smoke_b2_w8 | bf16 | 29.13 / 19.50 | 17.33 / 8.11 | 41.52 / 32.45 | 28.31 / 22.80 | 1.35 / 1.27 |
| smoke_b8_w24 | bf16 | 161.90 / 90.93 | 193.77 / 36.12 | 216.32 / 122.82 | 221.35 / 86.87 | 9.56 / 10.56 |
| main_b1_w24 | fp32 | 154.69 / 72.18 | 175.67 / 27.30 | 200.45 / 111.75 | 237.04 / 82.36 | 2.09 / 2.30 |
| main_b4_w24 | fp32 | 156.88 / 77.24 | 175.96 / 31.51 | 197.90 / 127.36 | 236.63 / 89.07 | 5.79 / 6.03 |

![Components](figures/main_components.png)

| Case | Precision | score graph nodes | CKY launches | full launches | score stack | score cat | score select | Python self ms | Composer rows/calls |
|---|---|---|---|---|---|---|---|---|---|
| main_b1_w24 | bf16 | 3031 / 892 | 4083 / 2210 | 14852 / 6276 | 1130 / 0 | 1216 / 88 | 1772 / 0 | 89.09 / 37.74 | 182 / 23 |
| main_b4_w24 | bf16 | 3031 / 892 | 4157 / 2284 | 14908 / 6332 | 1130 / 0 | 1216 / 88 | 1772 / 0 | 95.08 / 42.88 | 785 / 23 |
| real_b2 | bf16 | 2139 / 676 | 3144 / 1785 | 9683 / 4229 | 782 / 0 | 852 / 72 | 1236 / 0 | 70.47 / 28.59 | 243 / 18 |
| smoke_b2_w16 | bf16 | 1461 / 546 | 2335 / 1410 | 6681 / 3501 | 498 / 0 | 552 / 56 | 796 / 0 | 51.53 / 32.46 | 284 / 15 |
| smoke_b2_w24 | bf16 | 2961 / 822 | 4120 / 2247 | 13885 / 5309 | 1130 / 0 | 1216 / 88 | 1772 / 0 | 85.14 / 37.86 | 368 / 23 |
| smoke_b2_w8 | bf16 | 537 / 334 | 927 / 630 | 2607 / 1943 | 122 / 0 | 144 / 24 | 204 / 0 | 20.32 / 15.47 | 160 / 7 |
| smoke_b8_w24 | bf16 | 2961 / 822 | 4169 / 2296 | 13917 / 5341 | 1130 / 0 | 1216 / 88 | 1772 / 0 | 95.55 / 44.47 | 1373 / 23 |
| main_b1_w24 | fp32 | 2990 / 851 | 4062 / 1935 | 14250 / 5442 | 1130 / 0 | 1216 / 88 | 1772 / 0 | 93.25 / 36.67 | 182 / 23 |
| main_b4_w24 | fp32 | 2990 / 851 | 4127 / 2000 | 14370 / 5562 | 1130 / 0 | 1216 / 88 | 1772 / 0 | 118.12 / 43.55 | 748 / 23 |

Kernel为独立profiler的cudaLaunchKernel*计数。score节点含Syntax，另存isolated_score_autograd_nodes；Python self为cProfile项目函数exclusive wall time，非OS thread CPU或纯GPU时间。各配对实际Composer rows/calls相等；不能套用旧轮866行，当前BF16 main B4为785行/23次、FP32为748行/23次。

共享模型预热两套缓存后的peak，以及清空未用实现缓存的独立完整update显存审计：

| Case | Precision | shared-cache peak GiB | isolated active-cache peak GiB | isolated delta MiB |
|---|---|---|---|---|
| main_b1_w24 | bf16 | 2.7285 / 2.7285 | 2.7238 / 2.7229 | -0.89 |
| main_b4_w24 | bf16 | 2.7438 / 2.7438 | 2.7371 / 2.7409 | +3.81 |
| real_b2 | bf16 | 0.4084 / 0.4084 | 0.4052 / 0.4097 | +4.63 |
| smoke_b2_w16 | bf16 | 0.4141 / 0.4141 | 0.4118 / 0.4155 | +3.85 |
| smoke_b2_w24 | bf16 | 0.4170 / 0.4170 | 0.4130 / 0.4190 | +6.09 |
| smoke_b2_w8 | bf16 | 0.4141 / 0.4141 | 0.4113 / 0.4148 | +3.57 |
| smoke_b8_w24 | bf16 | 0.6642 / 0.6643 | 0.6465 / 0.6588 | +12.61 |
| main_b1_w24 | fp32 | 2.7254 / 2.7254 | 2.7222 / 2.7239 | +1.73 |
| main_b4_w24 | fp32 | 2.7613 / 2.7613 | 2.7502 / 2.7604 | +10.44 |

共享缓存下peak相同不代表零缓存成本。独立memory audit是每臂预热后的单次完整update，不混成8次普通计时的峰值。workload另有原DAG topology/index payload和静态缓存字节，payload不冒充peak。新DAG输入不产生大量Python Cell。

细分诊断各3次：Legacy独立source probe replay去掉探针行后逐字节等于原方法，单测证明输出、全部梯度和RNG不变；V3使用内置region。未重算Reader/Attention。表是各width求和后中位数，envelope含CPU launch gaps和观察开销，不能当纯GPU active time或完整加速比分母。

| Case | Precision | Arm | Gather+Pair ms | Selection ms | ID decode ms | Chart assembly ms |
|---|---|---|---|---|---|---|
| main_b1_w24 | bf16 | lazy_semantic | 84.35 | 31.56 | 7.46 | 15.99 |
| main_b1_w24 | bf16 | lazy_tensorized | 31.66 | 35.72 | 2.92 | 11.14 |
| main_b4_w24 | bf16 | lazy_semantic | 101.89 | 38.79 | 9.24 | 18.67 |
| main_b4_w24 | bf16 | lazy_tensorized | 30.06 | 37.20 | 3.11 | 11.23 |
| real_b2 | bf16 | lazy_semantic | 68.09 | 29.94 | 7.02 | 14.12 |
| real_b2 | bf16 | lazy_tensorized | 22.94 | 27.80 | 2.32 | 8.07 |
| smoke_b2_w16 | bf16 | lazy_semantic | 51.19 | 26.63 | 6.46 | 12.17 |
| smoke_b2_w16 | bf16 | lazy_tensorized | 17.55 | 21.99 | 1.77 | 7.00 |
| smoke_b2_w24 | bf16 | lazy_semantic | 103.51 | 39.04 | 9.22 | 19.45 |
| smoke_b2_w24 | bf16 | lazy_tensorized | 28.75 | 34.12 | 2.66 | 10.53 |
| smoke_b2_w8 | bf16 | lazy_semantic | 12.62 | 9.08 | 2.58 | 4.17 |
| smoke_b2_w8 | bf16 | lazy_tensorized | 7.10 | 9.00 | 0.85 | 3.39 |
| smoke_b8_w24 | bf16 | lazy_semantic | 105.41 | 37.96 | 9.01 | 20.05 |
| smoke_b8_w24 | bf16 | lazy_tensorized | 35.91 | 41.77 | 3.25 | 12.62 |
| main_b1_w24 | fp32 | lazy_semantic | 101.63 | 38.87 | 9.16 | 19.12 |
| main_b1_w24 | fp32 | lazy_tensorized | 25.27 | 37.18 | 2.96 | 10.99 |
| main_b4_w24 | fp32 | lazy_semantic | 113.06 | 41.98 | 10.00 | 19.96 |
| main_b4_w24 | fp32 | lazy_tensorized | 23.83 | 35.56 | 3.07 | 11.36 |

main B4 BF16全部width，有效候选数新旧一致；其他shape见CSV：

| Width | candidate capacity | valid candidates | envelope Lazy/V3 ms | launches Lazy/V3 |
|---|---|---|---|---|
| 2 | 368 | 368 | 3.09 / 2.28 | 43 / 39 |
| 3 | 704 | 704 | 4.13 / 3.12 | 52 / 43 |
| 4 | 1680 | 1680 | 4.57 / 3.15 | 73 / 59 |
| 5 | 3840 | 3840 | 4.98 / 3.23 | 79 / 60 |
| 6 | 6080 | 6080 | 5.67 / 3.50 | 91 / 67 |
| 7 | 9216 | 9216 | 5.59 / 3.55 | 97 / 68 |
| 8 | 13056 | 13056 | 5.90 / 3.70 | 103 / 69 |
| 9 | 16384 | 16384 | 6.39 / 3.65 | 109 / 70 |
| 10 | 19200 | 19200 | 6.45 / 3.47 | 115 / 71 |
| 11 | 21504 | 21504 | 7.92 / 3.51 | 121 / 72 |
| 12 | 23296 | 23296 | 7.07 / 4.07 | 127 / 73 |
| 13 | 24576 | 24576 | 7.75 / 3.78 | 133 / 74 |
| 14 | 25344 | 25344 | 7.40 / 3.68 | 139 / 75 |
| 15 | 25600 | 25600 | 8.47 / 3.68 | 145 / 76 |
| 16 | 25344 | 25344 | 8.07 / 3.89 | 151 / 77 |
| 17 | 24576 | 24576 | 8.32 / 3.81 | 157 / 78 |
| 18 | 23296 | 23296 | 9.01 / 3.57 | 163 / 79 |
| 19 | 21504 | 21504 | 8.83 / 5.09 | 169 / 80 |
| 20 | 19200 | 19200 | 9.45 / 4.46 | 175 / 81 |
| 21 | 16384 | 16384 | 8.87 / 3.79 | 181 / 82 |
| 22 | 13056 | 13056 | 10.25 / 3.90 | 187 / 83 |
| 23 | 9216 | 9216 | 9.43 / 4.90 | 193 / 84 |
| 24 | 4864 | 4864 | 11.04 / 3.73 | 192 / 82 |

![Widths](figures/width_envelope.png)

Profiler下单次完整update时间另列，不替代八次无profiler数据：

| Case | Precision | Arm | profiled wall ms | profiled CUDA envelope ms |
|---|---|---|---|---|
| main_b1_w24 | bf16 | lazy_semantic | 1057.2 | 1056.9 |
| main_b1_w24 | bf16 | lazy_tensorized | 467.0 | 466.9 |
| main_b4_w24 | bf16 | lazy_semantic | 1009.2 | 1008.9 |
| main_b4_w24 | bf16 | lazy_tensorized | 512.5 | 512.4 |
| real_b2 | bf16 | lazy_semantic | 645.3 | 645.1 |
| real_b2 | bf16 | lazy_tensorized | 306.4 | 306.3 |
| smoke_b2_w16 | bf16 | lazy_semantic | 526.4 | 526.2 |
| smoke_b2_w16 | bf16 | lazy_tensorized | 262.6 | 262.5 |
| smoke_b2_w24 | bf16 | lazy_semantic | 894.1 | 893.8 |
| smoke_b2_w24 | bf16 | lazy_tensorized | 391.0 | 390.9 |
| smoke_b2_w8 | bf16 | lazy_semantic | 258.3 | 258.1 |
| smoke_b2_w8 | bf16 | lazy_tensorized | 155.7 | 155.5 |
| smoke_b8_w24 | bf16 | lazy_semantic | 880.4 | 880.3 |
| smoke_b8_w24 | bf16 | lazy_tensorized | 434.9 | 434.8 |
| main_b1_w24 | fp32 | lazy_semantic | 954.5 | 954.3 |
| main_b1_w24 | fp32 | lazy_tensorized | 458.3 | 458.1 |
| main_b4_w24 | fp32 | lazy_semantic | 996.1 | 995.8 |
| main_b4_w24 | fp32 | lazy_tensorized | 479.8 | 479.6 |

## 8. 分阶段收益

A先完成60项结构/梯度专项和独立A/Lazy测速，随后实现B，完成完整矩阵/梯度/恢复测试，再做最终配对。A冻结源码位于baseline/；B/A增量也单独同轮配对，不跨轮相除。

| Comparison | Case | same-round update ms | paired ratio median | sample ratio range |
|---|---|---|---|---|
| A / Lazy | main_b4_w24 | 560.42 / 338.81 | 1.706x | [1.487,1.796] |
| A / Lazy | smoke_b2_w24 | 448.68 / 235.78 | 1.894x | [1.687,2.042] |
| B / A | main_b4_w24 | 333.26 / 309.69 | 1.047x | [1.017,1.146] |
| B / A | smoke_b2_w24 | 238.53 / 214.00 | 1.107x | [1.029,1.203] |

C未实施：A主模型隔离score backward已从173.1降到32.0ms，优先保留原可微图。未运行的score reconstruction、Triton、Compile或CUDA Graph没有填入收益表。

## 9. 剩余瓶颈

仍有跨width顺序、只读bank cat和反向分发、原Top-K/探索/Mask小kernel、一次拓扑D2H/CPU闭包/H2D、Composer小批及Decoder/Optimizer。D2H阻塞包含等待前序GPU，不能全归为DFS，组件不可叠加。PyTorch张量化已改善完整更新，尚无证据要求立刻引入Triton/compile；后续应独立检查剩余提交空隙再决定。本轮未使用上述技术，也未改变神经公式、K/R/Delta、探索分布、数据、Loss、LR或优化器。

## 10. 最终建议

有界K/R搜索身份、Mask和RNG保持；FP32数学/原严格梯度通过，BF16相对旧Lazy的这些case也通过。主模型BF16 B4组件中位数比：Score Forward **2.05x**、Score Backward **6.21x**，Forest Forward **1.25x**、Forest Backward **2.34x**；完整Update同repeat配对比中位数 **1.831x**，完整CUDA launches减少 **57.53%**。组件诊断与完整更新分别测量，不互相叠加。PyTorch Tensorization值得作为后续研究实现，正式默认保持score_first。

本地工程条件支持受控早期预训练试验；仍需目标硬件、更长序列、更大数据与多seed验证，20步不能证明预训练质量等价。可在Blackwell独立运行相同脚本并使用新输出目录，当前没有Blackwell实测或速度外推。没有自动进入编译或自定义kernel。

精度：BF16 outer、FP32 IEEE Composer、BF16 Reader、score_dtype existing。全FP32使用现有outer_autocast: fp32关闭AMP，未新增disabled别名。configs/v3.yaml为smoke，v3_main_125m.yaml为主模型，v3_fp32.yaml为IEEE对照。

复现，顺序执行，测速时不并发其他GPU任务：

```powershell
& 'E:/anaconda3/envs/py312/python.exe' validate_cky_v3.py
& 'E:/anaconda3/envs/py312/python.exe' train_cky_v3_validation.py
& 'E:/anaconda3/envs/py312/python.exe' train_cky_v3_validation.py --cuda-only --micro-batch-size 1 --output-dir reports/cky_v3/accumulation_training
& 'E:/anaconda3/envs/py312/python.exe' benchmark_cky_v3.py
& 'E:/anaconda3/envs/py312/python.exe' benchmark_cky_v3.py --precision fp32 --cases main_b1_w24 main_b4_w24 --resume
& 'E:/anaconda3/envs/py312/python.exe' benchmark_cky_v3.py --implementation lazy_tensorized_a --cases smoke_b2_w24 main_b4_w24 --output-dir reports/cky_v3/stage_a
& 'E:/anaconda3/envs/py312/python.exe' benchmark_cky_v3.py --reference lazy_tensorized_a --cases smoke_b2_w24 main_b4_w24 --output-dir reports/cky_v3/stage_b_increment
& 'E:/anaconda3/envs/py312/python.exe' profile_cky_v3_widths.py
& 'E:/anaconda3/envs/py312/python.exe' summarize_cky_v3.py
```

CUDA恢复脚本默认使用本机已有V2.5-P B2 seed42/500步checkpoint，大权重不打入源码包；可用--checkpoint指定可信Lazy检查点。公共小缓存在precision_study_cache.zip。更换设备/参数使用独立output-dir，不能resume混入本机测量。completion.json/artifact_hashes.json记录覆盖和校验。
