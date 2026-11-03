# 实际验证记录

验证日期：2026-10-31（Asia/Shanghai）。环境详细版本见 `validation_environment.json`。这是软件正确性与集成验证记录，不是正式预训练研究结果。

## 已实现并实际执行

- 完整源码、三种配置、全部六种训练模式、Forest Top-1 评估、Colab Notebook。
- 首次 CPU 环境的 `python -m pytest -q`：**38 passed, 1 skipped**。该环境使用 `torch 2.12.1+cpu`，唯一 skip 是 CUDA/BF16 前向反向测试。后续在用户的 py312 CUDA 环境中继续验证，见下面的补充记录。
- 长度 3–6 的 CKY 在关闭所有剪枝/预算后，与独立枚举的 2、5、14、42 棵树的集合、分数和系数一致。
- 测试覆盖非对称组合、Padding/空历史、归一化、剪枝、真正进入保留 beam 的探索、硬选择梯度限制、标量先验的 LM 梯度、未来 Token 因果性及历史编码边界。
- 普通无记忆 LM 在固定小批次上优化 25 次后 Loss 下降；此项是学习能力回归测试，不是泛化评估。
- 模型、优化器、调度器、Python/NumPy/Torch RNG、采样游标与句内 Token 偏移均做了恢复检查。包含 Dropout 的恢复更新与连续更新逐参数完全一致；完整训练入口换目录恢复也与连续训练一致，且保留日志历史。
- 正式 smoke 维度：Decoder 10,093,440 参数，完整向量森林 22,402,211 参数。合成缓存上完成 8192 个 CE 目标 Token，再从实际检查点恢复到 12288 Token。
- 六个模式均在相同合成数据缓存上实际完成 256 个 CE Token 的训练、保存、完整开发集评估，输出 `cpu_validation/matrix_summary.csv`。这些极小 CPU 运行不是公平吞吐量基准：期间有其他验证进程运行，不能据此给模型排序或外推 GPU 性能。
- 官方 TinyStories 的 train 前 32 个故事、validation 前 8 个故事，实际联网下载并完成独立词表/BPE、分句、故事级开发/测试划分及缓存。得到 train 442 句/5653 目标 Token、dev 63 句/1041 Token、test 59 句/805 Token。额外完成森林 256 Token 训练与开发集前两个 batch 的评估（265 Token）。该子集 Tokenizer 不是正式全量训练集的 16K Tokenizer。
- 实际执行正常 Memory、Memory Off、跨故事 Shuffled Memory、Top-1、前 8/后续 Token NLL、候选诊断、按故事 Bootstrap、模板条件似然、CSV 和 PNG 生成。
- 三种配置的所有模块参数数量已实际按真实参数形状计数，见 `smoke/parameters.csv`、`main_125m/parameters.csv`、`scale_245m/parameters.csv`。
- Notebook 的 nbformat schema 与全部 Python 单元的语法通过验证；Notebook 没有嵌入虚构的运行输出。

## 尚未实际验证或运行的范围

- 没有在 NVIDIA RTX PRO 6000 Blackwell 或 Google Colab hosted runtime 上执行。RTX 3070 Laptop 上已实际验证 CUDA/BF16、显存与小批次吞吐量，详见 `CUDA_PERFORMANCE.md`；Blackwell 和 Colab 环境仍需独立实测。
- 没有下载/预处理完整 TinyStories，也没有运行 25M、100M 或 500M Token 训练；没有正式多个 Seed 的统计结论。
- 没有覆盖 5M–100M 的真实曲线，因此不提供正式 AULC 或达到阈值成本结果。积分和阈值代码由构造数据验证。
- Google Drive 的实际挂载与远程同步未在本机验证。实现了本地原子检查点及远程临时复制/rename；最终原子性由挂载文件系统决定。
- 原生 SDPA、按区间长度批处理的 CKY 可直接运行，无自定义 CUDA 编译。后续 CUDA 剖析已确认 Python chart 调度、张量重组及 CKY backward 的大量小节点构成主要瓶颈，详见 `CUDA_PERFORMANCE.md`。
- 分句为启发式，长历史/目标按 README 截断。模板测试仅用于探索，不能代替人工语义基准或 Gold Parse 指标。

`cpu_validation/` 内的 NLL、时间、梯度、置信区间和图均来自上述真实小规模运行，只用于确认管道工作。GPU 小时和显存字段为空。主配置的 Decoder 为 125,793,792 参数，完整向量森林为 148,840,709 参数；扩展配置相应为 243,249,152 和 273,181,574。

## 后续 py312 CUDA 验证

用户提供的 `E:\anaconda3\envs\py312\python.exe` 为 Python 3.12.13、PyTorch 2.12.1+cu132。RTX 3070 Laptop 8GB 上实际执行了 BF16 森林的前向、反向、非零语法先验/Composer/Reader 梯度检查，以及主配置和 smoke 的性能剖析。小批次数据、计时边界和限制均保存在 `CUDA_PERFORMANCE.md` 及 `cuda_profile/`。

修改后的完整测试在 py312 环境中实际得到 **39 passed in 9.19s**，CUDA 测试没有跳过。命令：`python -m pytest -q --tb=short --basetemp=.pytest_tmp_py312 -o cache_dir=.pytest_cache_py312`；该独立临时路径用于避开此前 CPU 沙箱留下的目录权限。

加入 tqdm 后，在真实 TinyStories 小缓存上实际训练 128 个 CE Token，再从检查点恢复到 256；进度条从 128/256（50%）开始，Token 计数、模型/优化器恢复和 CSV/JSONL 输出正常。原始指标副本见 `cuda_profile/training_progress_metrics.csv`。该小训练验证不提供正式预训练研究结论。

## CKY V2 验证补充

V2 完成后，py312 全套测试为 **205 passed in 55.01s**。额外验证包含 576 组 FP32/BF16、训练/评估的每 Span 候选及 LM 前向对比，32 个边界案例，6 个完整 LM 梯度对比与两条 8 步训练轨迹。新旧实现选择、公共非有限势修复、低精度梯度误差及中性 scalar bias 漂移均明确记录在 [V2 报告](cky_v2/summary.md)。

在同一份模型/AdamW/RNG 快照上重新成对测量六种指定形状和真实动态 Padding 批次；不是沿用历史 GPU 数据。原始 V0 源码、修改前测量和全部新结果均在 `cky_v2/`。没有改动模型架构、K/R、数据、优化器或实现 V3。


## CKY V2.5（2026-10-31）

247 tests passed、1 skipped；1080 个 shape/device/mode 矩阵和 48 个边界例；8 组梯度对照，CPU FP32 8 步与 CUDA BF16 20 步双模式训练。主模型 FP32 通过；BF16 原严格梯度容差未通过，默认仍为 V2。已完成 7 组 BF16 和额外 main B1 FP32 的同权重/AdamW/RNG 成对测速，每侧 warmup 2、采样 5。详见 [V2.5 完整报告](cky_v25/summary.md)、[验收状态](cky_v25/validation_protocol.json)。


## CKY V2.5-P（2026-10-31）

完成 BF16/FP32 IEEE/FP32 TF32/FP16 Composer 与全 FP32 正控制、独立 score FP32 补充。1440 个正向矩阵 + 50 边界例；12 组主模型/Scalar梯度配对及额外 10 组 B4/W24 完整模型梯度配对；282 passed、1 skipped。35 对/70行性能记录、每侧 warmup2+采样5。6种方案100步筛选，B1/B2/C2各500步×种子13/42/2026；真实固定子集、相同CE batch、完整固定dev。建议后续优先B2，默认保持V2；没有进入V3。BF16全模型严格梯度失败仍如实记录。

详见 [完整研究报告](cky_v25_precision/summary.md)、[最终决策](cky_v25_precision/recommendation.json)、[覆盖与完整性](cky_v25_precision/completion.json)。只实测RTX3070 Laptop，提供Colab入口，不伪造Blackwell结果。


## CKY V2.5-R（2026-10-31）

py312 CUDA 全套 **288 passed, 1 skipped in 58.57s**；skip 为单词句不存在二词子区间的既有结构测试，CUDA Reader 测试均执行。新增六项测试覆盖 precision/参数兼容、三种精度下只读 native SDPA probe（输出/输入与参数梯度/RNG逐位一致）、可微返回 dtype 边界及旧 checkpoint reader 默认字段兼容。

main_125m batch1/4 的 B/R/E 各执行一组完整 CE forward/backward 配对（12次完整运行），同权重/输入/RNG、Dropout 开启；逐 span Candidate ID/Backpointer 与 Reader 入口 RNG 审计通过。六组无梯度缓存输入控制均输出完全一致，但不作为 backward 等价证据。Reader 改 FP32 不改变各实现根语义；全模型梯度误差降低9.39%/6.67%，BF16 严格失败数由27/7降至24/6，E正控制0失败；所有完整梯度非有限数为0。

五种形状同轮 B1/B2/B1-R/B2-R 性能20行，各 warmup2+有效8样本；组件诊断、dtype与独立profiler重放单独保存。没有改动搜索、模型结构、参数数量、数据、优化器或学习率；无额外训练及V3。详见 [Reader 隔离完整报告](cky_v25_reader/summary.md) 与 [完整性审计](cky_v25_reader/completion.json)。

## CKY V3（2026-10-31）

全套 **413 passed, 1 skipped in 76.41s**。1080 结构矩阵+36边界，10组 CPU/CUDA FP32/BF16 完整模型梯度配对均通过旧严格容差；逐Span候选身份、回溯、分数、Mask及所有 RNG 一致，完整LM前向逐位一致。原Lazy DAG尾部逐字节共享，神经模块源码/参数保持；不使用 no-grad 搜索或 score reconstruction。

CPU Scalar/Vector各8步；真实TinyStories smoke模型从B2 seed42/500步恢复到520，8000到8320CE tokens，V3第510→511步checkpoint重放的参数/Optimizer/RNG/scheduler/sampler逐位一致。另以microbatch1、effectiveCE16测20步，实际每次更新2–3个microbatch。权重与数值score漂移完整记录，不声称训练轨迹逐位等价。

九组最终GPU配对，每臂warmup2+有效8样本；另A/Lazy与B/A各两组独立同轮对照。逐width有效数/Kernel/选择/Gather/装配、score/forest F/B、profile与普通整步、共享/独立缓存峰值分别记录。无界K=None且R=None的额外严格梯度探针初次失败后保留原搜索兼容路径，未放宽容差或删除失败测试。默认score_first未改，未使用Triton/Compile/CUDA Graph。详见 [完整V3报告](cky_v3/summary.md)、[覆盖与交付校验](cky_v3/completion.json)。

## 顶层训练 Notebook（2026-11-01）

新增根目录 `Train.ipynb`，19个单元，复用原 `train.run()` 与评估代码。通过 nbformat schema、全部代码单元语法检查；使用 nbclient 启动真实 py312 CUDA 内核执行，V3 smoke/真实小缓存先训练512 CE Token，再通过 Notebook追加恢复到768，步数[1,2,3]、学习率与调度计数连续。训练曲线和开发集前两个batch评估（116目标Token）实际完成。另验证顶部恢复配置保留checkpoint中的模型/Seed/effective tokens，忽略新训练参数，累计预算单独提高。

交付Notebook执行计数与输出均清空；原始执行副本在`runs/notebook_validation/executed_train.ipynb`，实际环境、路径和断言记录见 [Notebook 验证记录](notebook_training_validation.json)。本次没有修改模型、优化器或训练脚本，也没有重新进行研究Benchmark。


## Local CNN memory baseline (2026-11-03)

Added `seq_cnn4`, masked token-local CNN blocks, kernel=1/3 controls and `local_cnn_ablation.py`. Related model, data, resume, precision and Reader checks: **53 passed**. CUDA smoke trained on the existing tiny real-data cache to 256 CE tokens, resumed through the original training CLI to 384, and evaluated the 256-token checkpoint on the entire tiny-cache dev split (1041 targets). Notebook code is unchanged; CNN usage documentation was added and schema/AST/cleared outputs validated. Resident-batch full-update timings and scope limitations are documented in [local CNN report](local_cnn/README.md). No formal architecture-quality comparison was performed.


## Portable training notebook (2026-11-03)

`build_train_notebook.py` generates the Git-ignored, standalone `Train.ipynb`, with clone/install/authentication and source-independent cache/output paths. Seven launcher tests passed, including actual local Git clone, separate storage, secret handling and cache archive relocation. A full CUDA notebook execution from a fresh local clone trained 512 CE tokens and resumed to 768; plots and 116-token dev evaluation completed. Colab Secrets and hidden-input paths are unit-tested with test credentials; live Colab, network dependency installation and Google Drive mounting were not exercised. No plaintext HF token is embedded. Details: `notebook_portable_validation.json`; logs: `notebook_portable/`.
