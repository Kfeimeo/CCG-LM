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

{cache}

正式计时已预热，固定shape缓存命中；累计hit rate包含首次miss。build是主机规划/H2D提交时间，非纯GPU active time。

## 4. Candidate Selection

原生topk保留同compact形状、顺序及末维stride，无新ID排序。执行顺序仍为Top-R预算拆分、proposal探索、可微score gather、Top-K、末槽探索替换、Delta。三个rand_like按原width顺序和相同形状/dtype执行；Python/NumPy/CPU/CUDA RNG逐项核对，不依赖旧探索重放。

B保留已经batched的选择，只优化ID解码、Cell/root装配。完全同分/近同分、欠满K/R、无效槽及Delta=0均测试。唯一兼容分支为额外的K=None且R=None精确Catalan枚举：近零score梯度累计曾差7.629e-6，超过严格3e-6/3e-4。保留失败测试和logs/v3_b_unbounded_strict_failure.log，使用原Lazy搜索保持其梯度，不放宽标准。正式K=1/2/4、R=4/12仍走张量化。

## 5. Autograd 正确性

选中score和root log-prior保留到SpanScorer的可微引用；仅原离散priority detach。独立prior backward验证Scorer非零。共享DAG、空闭包的原Composer零梯度anchor保持，未冻结中性bias。

`_finish_lazy` 提取旧尾部；拓扑D2H、CPU闭包/memoization、plan上传、width Composer、根语义/系数组装主体与冻结源码逐字节相同，见source_scope_audit.json。冻结原实现和提取后的输出/输入与参数梯度逐位回归通过。神经模块源码/参数名称/数量保持；没有no-grad搜索或score reconstruction。

## 6. Forward / Backward 验证

1080矩阵+36边界：B=1/2/4，W=1/4/8/16/24，K=1/2/4，R=4/12，Scalar/Vector，CPU FP32、CUDA IEEE FP32、BF16 outer，Eval/Train探索。逐Span比较metadata、selected/retained flat、左右ID、base、score/mask及树次序；根score/prior/coeff/semantic和完整LM前向逐位相同。BF16测试score/coeff实际保存为BF16。无历史/无效history另有模型单测，原score_first/Lazy交叉检查也保留。

{gradient}

旧容差不变：FP32 atol=3e-6/rtol=.0003，BF16 atol=.00015/rtol=.04。10组完整LM零严格失败，此结论针对旧Lazy与V3，不覆盖上一轮V2/Lazy已有BF16差异。前向max abs/L2/RMS在JSON，七模块范数和显著梯度误差在CSV；vector head是span scorer子集，全模型不重复计数。

{modules}

{training}

CPU FP32每模式8步同batch/RNG；真实CUDA从旧B2 seed42/500步checkpoint续训至520步，8000到8320 CE tokens，优化器/LR/数据指纹连续。GPU重放验证两臂统一确定性算法，与普通benchmark设置区分。V3第510步保存后第511步重放，参数、Optimizer矩、RNG、scheduler和sampler逐位一致。短训练仍有权重漂移，不冻结中性bias，不以20步证明预训练质量等价。

轨迹中的search/probe_search沿用旧helper的合计比较，包含数值score；不能由非零changed_entries直接认定Candidate ID或树结构改变。固定权重的身份一致性在矩阵中独立验证，训练过程没有声称逐步参数或搜索统计完全相同。

真实默认microbatch4的20步均只用1个microbatch，Lazy/V3为{real_lazy}/{real_v3} CE/s。另以相同effective CE=16、显式microbatch1独立跑20步，实际每步micro_batches={micro}，Lazy/V3为{acc_lazy}/{acc_v3} CE/s。后者确实含mmap取样、Collator、H2D、多microbatch梯度累积、clip和AdamW；排除保存和独立probe，不与驻GPU微批吞吐混称。microbatch切换沿用恢复接口警告，两臂共享分块/输入/RNG，effective预算、LR和优化器未变。详见training/与accumulation_training/。

![Training](figures/training.png)

回归：`{tests}`。初始诊断Hook布尔参数异常及无界严格梯度失败日志保留。

## 7. GPU 性能

RTX3070 Laptop8GB，torch2.12.1+cu132，CUDA13.2。七种BF16形状+两种全FP32主模型，每侧warmup2、有效8样本，交替顺序、恢复同一物理Model/AdamW/scaler/RNG；共同零梯度AdamW buffers setup含相同weight decay，setup/恢复不计时。输入驻GPU，计时含CE forward/backward/clip/AdamW。真实batch H4/W20/T17、CE29，数据未变。完整min/max、组件分布、输入/权重hash均保留。

{full}

![Full update](figures/full_update.png)

以下独立组件各3次，单位ms，不可相加还原整步。Score-only backward使用完整LM真实log-prior边界梯度，两臂相同；独立Span势作为叶子仅用于隔离诊断，正常完整更新没有detach。

{parts}

![Components](figures/main_components.png)

{ops}

Kernel为独立profiler的cudaLaunchKernel*计数。score节点含Syntax，另存isolated_score_autograd_nodes；Python self为cProfile项目函数exclusive wall time，非OS thread CPU或纯GPU时间。各配对实际Composer rows/calls相等；不能套用旧轮866行，当前BF16 main B4为785行/23次、FP32为748行/23次。

共享模型预热两套缓存后的peak，以及清空未用实现缓存的独立完整update显存审计：

{memory}

共享缓存下peak相同不代表零缓存成本。独立memory audit是每臂预热后的单次完整update，不混成8次普通计时的峰值。workload另有原DAG topology/index payload和静态缓存字节，payload不冒充peak。新DAG输入不产生大量Python Cell。

细分诊断各3次：Legacy独立source probe replay去掉探针行后逐字节等于原方法，单测证明输出、全部梯度和RNG不变；V3使用内置region。未重算Reader/Attention。表是各width求和后中位数，envelope含CPU launch gaps和观察开销，不能当纯GPU active time或完整加速比分母。

{fine}

main B4 BF16全部width，有效候选数新旧一致；其他shape见CSV：

{width}

![Widths](figures/width_envelope.png)

Profiler下单次完整update时间另列，不替代八次无profiler数据：

{profile}

## 8. 分阶段收益

A先完成60项结构/梯度专项和独立A/Lazy测速，随后实现B，完成完整矩阵/梯度/恢复测试，再做最终配对。A冻结源码位于baseline/；B/A增量也单独同轮配对，不跨轮相除。

{stages}

C未实施：A主模型隔离score backward已从173.1降到32.0ms，优先保留原可微图。未运行的score reconstruction、Triton、Compile或CUDA Graph没有填入收益表。

## 9. 剩余瓶颈

仍有跨width顺序、只读bank cat和反向分发、原Top-K/探索/Mask小kernel、一次拓扑D2H/CPU闭包/H2D、Composer小批及Decoder/Optimizer。D2H阻塞包含等待前序GPU，不能全归为DFS，组件不可叠加。PyTorch张量化已改善完整更新，尚无证据要求立刻引入Triton/compile；后续应独立检查剩余提交空隙再决定。本轮未使用上述技术，也未改变神经公式、K/R/Delta、探索分布、数据、Loss、LR或优化器。

## 10. 最终建议

有界K/R搜索身份、Mask和RNG保持；FP32数学/原严格梯度通过，BF16相对旧Lazy的这些case也通过。主模型BF16 B4组件中位数比：Score Forward **{score_f}x**、Score Backward **{score_b}x**，Forest Forward **{forest_f}x**、Forest Backward **{forest_b}x**；完整Update同repeat配对比中位数 **{update_speedup}x**，完整CUDA launches减少 **{kernel_reduction}%**。组件诊断与完整更新分别测量，不互相叠加。PyTorch Tensorization值得作为后续研究实现，正式默认保持score_first。

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
