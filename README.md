# Latent Semantic Forest Language Model

一个从零训练的小型语言模型研究项目：完成的历史句子先经过潜在连续二叉树组合，得到多个带权语义候选；GPT Decoder 通过 Semantic Cross-Attention 预测下一句。无 CCG 范畴、人工句法标签或 Lambda Term。

完整实验入口是 `colab_experiment.ipynb`。源码使用 Python 3.11+、PyTorch 2.x 和原生 SDPA，不需要编译自定义 CUDA 扩展。Blackwell 上请使用支持实际 GPU 架构的 CUDA 版 PyTorch；Notebook 会运行真实 CUDA/SDPA 运算检查，不能只依据 `nvidia-smi` 判断可运行性。

## 快速开始

在项目根目录执行。Windows 可用 `.venv/Scripts/python.exe` 替换下方的 `python`。

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python -m data.prepare --config configs/smoke.yaml --demo
python train.py --config configs/smoke.yaml --output-dir runs/smoke
python evaluate.py --checkpoints runs/smoke/tokens_8192.pt --cache-dir cache/demo --output-dir runs/smoke/evaluation --diagnostics --top1 --logs runs/smoke/metrics.csv
python semantic_tests.py --cache-dir cache/demo --checkpoint runs/smoke/tokens_8192.pt
```

`--demo` 使用合成故事，只验证工程流程，不能用于报告 TinyStories 研究结论。默认 smoke 的总预算为 8192 个 CE 目标 Token。首次下载真实数据必须省略 `--demo` 并选择不同缓存目录。

```bash
python -m data.prepare --config configs/main_125m.yaml
python train.py --config configs/main_125m.yaml --mode forest_vector4 --seed 42 --output-dir runs/main_vector_seed42
python train.py --config configs/main_125m.yaml --mode forest_vector4 --seed 42 --output-dir runs/main_vector_seed42 --resume runs/main_vector_seed42/tokens_25000000.pt --budget-tokens 100000000
```

主配置默认停在 **25M**；参考余弦终点始终是 **500M**，恢复到 100M 时不重启 warmup，不改变调度轨迹。5M、10M、25M、50M、100M 处均精确按 CE Token 保存，任何一次训练预算的终点也会保存。

## 模块、形状与数据流

```text
data.prepare -> mmap cache -> StoryDataset/Collator
                             | completed history [B,H,W]
                             v
                Syntax Embedding -> bidirectional SyntaxEncoder
                             | local span scalar / optional vector
                Semantic Embedding -> CKY + ordered Composer
                             v
           semantic [B,H,K,D], log_prior [B,H,K], coeff [B,H,K,C]
                             | candidate_mask [B,H,K], history_mask [B,H]
                             v
target BOS + prefix [B,T] -> causal Decoder + SemanticReader -> logits [B,T,V]
labels [B,T] (EOS included, padding=-100) ------------------> CE sum / true tokens
```

主配置 `H<=4,W<=24,T<=128,K<=4,D=256,C=32,V=16000`。`SpanScorer` 张量下标 `[i,j-1]` 表示半开区间 `[i,j)`。标量版本内部系数维度为 0，向量版本为 32；Reader 采用相同字典接口，只在向量版本使用系数投影参数。所有历史版本在 Reader 前补齐到四个槽位并用 Mask 标注真正合法的候选。

Syntax 与 Semantic Embedding 各为独立的 256 维参数。Syntax Encoder 的整句状态仅产生局部结构势与系数，绝不直接作为 Composer 的语义输入。目标句子没有进入 Syntax Encoder 的路径。

## 数据与缓存

- 官方源：[roneneldan/TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)，字段 `text`。以 streaming 下载到本地 JSONL，重启后复用。
- 官方 train 单独拟合 Byte-level BPE 和词级词表；BPE 目标 16K、词级最多 20K，二者 Token ID 独立。小型 demo 的实际词表可能更小；模型配置保留词表容量上限。
- 官方 validation 使用 seed=2026 的故事 ID 随机排列，一半开发、一半最终测试；之后各自保持原始故事顺序。任何训练样本不跨故事边界。
- 简单英文分句器保护常用缩写、缩合词、句尾引号与小数，属于启发式分句，不是人工标注。词级切分保留标点，统一小写和弯引号缩合形式。
- 每个句子都能成为目标，包括故事首句（空历史）。历史为之前最多四个已完成句子；超过 24 词的历史句子保留前 24 个词/标点 Token。目标超过上限时保留前 127 个 BPE Token，再加 EOS。词级完整句子索引仍然保持故事边界。
- `.bin` Token 数据用 mmap 读取，`.npy` 保存句子/故事偏移。缓存包含词表、Tokenizer、划分、SHA256 与 manifest；manifest 最后写入作为完成标记。原始文本及 Token 数据只需处理一次。
- 配置或原始源选择变化时拒绝静默复用缓存。用新 `cache_dir` 建立新缓存。`--verify-cache` 可在复用时校验所有 Token 文件 SHA256。缓存指纹不包含绝对目录，因此可在本地与 Drive 间移动。

可限制故事数做真实数据集成检查（请使用独立 YAML 缓存目录）：

```bash
python -m data.prepare --config configs/main_125m.yaml --max-train-stories 1000 --max-validation-stories 100
```

后续复用此缓存时保持相同的两个 limit 参数；正式训练使用全量缓存，不要混用该子集训练出来的 Tokenizer。

## CKY 与梯度约定

叶子包含该词的 Semantic Embedding 和叶 Span 的结构势；非叶候选计算：

```text
h_parent = Composer(h_left, h_right)
score_parent = score_left + score_right + s[i:j]
u_parent = u_left + u_right + u[i:j]
```

Composer 是共享的有序拼接 MLP（隐层 512）加左右独立残差投影、Sigmoid 门控和 LayerNorm。不存在对称性约束。

生产搜索按长度批处理所有句子和同长度区间。每个 Span 先按累积结构势选择至多 12 个待组合候选，执行神经组合，保留至多 K=4，应用相对最优分数 Delta=8 的剪枝。训练探索会在预算内抽取额外候选，并以配置概率让非最佳候选进入保留 beam，始终保留最佳合法候选。每个合法 Span 至少一个候选。

默认 `model.forest_implementation: score_first` 使用 V2：只在标量分数/Mask 上进行 Top-R 初筛，随后解码真实 split/左右槽位，从临时只读子候选 bank 中 gather 被选中的语义与系数。Cell 字典、逐 split 循环、K/R、探索和 Delta 顺序均保留；没有引入完整 Tensorized Chart、compile 或自定义 Kernel。`reference` 保留筛选前展开高维配对的路径。两条路径共享相同搜索代码；非有限局部势显式标记不可达，避免全 `-inf` 根先验 NaN。

```bash
python train.py --config configs/smoke.yaml --forest-implementation score_first
python train.py --config configs/smoke.yaml --forest-implementation reference --output-dir runs/reference
python profile_performance.py --config configs/main_125m.yaml --batch-size 1 --lengths 24 --forest-implementation score_first
python validate_cky_v2.py --output-dir reports/cky_v2
python benchmark_cky_v2.py --output-dir reports/cky_v2 --warmup 2 --repeats 5
python -m pytest tests/test_forest_v2.py -q
```

`benchmark_cky_v2.py` 在每次样本测量前恢复同一份权重、AdamW 和 RNG 快照，交替测量两条实现；只读快照恢复与 profiler 均不计入普通计时。验证脚本比较各 Span 的候选身份和顺序、输出、完整模型梯度及 8 步训练轨迹。Checkpoint 的参数键不变，恢复时可用 `--forest-implementation` 切换；浮点累计顺序存在微小梯度差异，不承诺跨实现逐位一致。完整 V2 结果与限制见 `reports/cky_v2/summary.md`；先前性能报告记录的是 V0。

V2.5 可通过 `--forest-implementation lazy_semantic` 显式启用。它保留可微结构分数图，完成全部搜索后，只对有效根闭包内的 Candidate ID 构建共享 DAG，再按跨度宽度批量调用原 Composer。搜索期的 `ScoreCell` 不持有高维语义/系数；离散回溯信息只进行一次批量 D2H，传输同步与 CPU 建图时间单独计量。默认仍为 `score_first`，三种实现的参数键和 checkpoint 格式一致。

```bash
python train.py --config configs/smoke.yaml --forest-implementation lazy_semantic --output-dir runs/lazy
python validate_cky_v25.py --output-dir reports/cky_v25
python benchmark_cky_v25.py --output-dir reports/cky_v25 --warmup 2 --repeats 5
python -m pytest tests/test_forest_lazy.py -q
```

当前 V2.5 的 FP32 检查通过；主模型 BF16 未通过 V2 原严格逐元素梯度容差，20 步训练也记录到数值漂移，因此维持实验开关与 V2 默认路径。全量验证器会写完报告后对严格梯度不匹配返回非零退出码。V2.5 的实际数值误差、多步训练漂移、成对速度和内存结果见 [V2.5 报告](reports/cky_v25/summary.md)。这里只复用同一个模型的确定性 Composer；训练态随机 Composer 或 BatchNorm 会被拒绝。CPU 建图与按宽度的小 GEMM 都可能抵消计算量收益，因此保留 V2 路径供对照。

**12 是近似搜索预算。** 有限 beam、预算和阈值都会丢弃推导；不是对全部树的精确求解。短句测试可设 `k=None,max_compositions=None,delta=None,exploration=0`，枚举全部连续二叉树并与独立枚举器核对数量、势分数、树和系数。

Span 标量是局部 log-potential，不是全局归一化树概率。根部 softmax 只在本句**保留**的候选中归一化。保留分数通过 Reader 的 `beta * log_prior` 接受 LM 梯度。硬 Top-K、预算选择、随机结构选择与阈值本身不可微，未保留候选不会因硬选择获得普通反向传播梯度。

`forest_top1_eval` 只能从训练好的森林执行评估。单候选先验恒为 1，标量 Scorer 无法从这个路径训练；测试明确覆盖此限制。向量系数路径与标量先验路径分别实现，标量模式测试证明不是依赖向量头才收到梯度。

## Semantic Reader 与因果性

Decoder 使用 learned positions、Pre-LN、GELU FFN、共享输入/输出嵌入和原生 [PyTorch SDPA](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)。主配置在第 4、8、12、16 层后读取历史。Reader Key 来自语义及可选系数，Value 仅来自语义，注意力偏置为句内 `log_prior` 和距目标句的句间位置偏置；同一句四个候选共享同一个位置。

Adapter 初始门控为 `sigmoid(-2)≈0.119`，足够小且初始即可传梯度。空历史使用安全 sentinel，输出强制为零，避免全 Mask 的 softmax NaN。`memory_off=True` 直接跳过历史编码和 Reader，输出不依赖任何历史张量。

Reader 在需要对加性 Bias 求导时显式使用原生 SDPA math backend，保证语法先验和句间偏置的梯度，不依赖特定 fused Kernel 对 Mask 梯度的支持。Decoder 的因果自注意力仍允许 PyTorch 自动选择优化 backend；无梯度评估也允许自动选择。Reader 的记忆长度最多 16，math 路径的注意力矩阵很小。

目标输入是 `BOS + target[:-1]`，标签是 `target + EOS` 的适当截断序列。Decoder 只有因果自注意力；右 Padding 永远不会出现在有效 Query 的过去，所以不会改变有效位置输出。测试会修改未来目标 Token，并检查之前 Logits 完全不变。

## 对照组及参数

`--mode` 支持 `no_memory`、`pooling4`、`fixed_tree4`、`random_tree4`、`forest_scalar4`、`forest_vector4`。Pooling 是四个可学习 Query 对 Semantic Embedding 的注意力池化；固定树为左分支、右分支、平衡和交替分支；随机树按 Catalan 数加权采样，句子内容和实验 Seed 决定结构，合法结构足够时采样四棵不同树，短句重复槽位以保持 Reader 容量。

Decoder 在同一配置的各版本完全一致。树/池化对照没有添加闲置 Syntax 参数；容量差异如实报告。不同参数组的真实计数可通过 meta device 无大模型分配地计算：

```bash
python experiments.py --config configs/main_125m.yaml --parameters-only --output-dir reports/main_125m
python experiments.py --config configs/scale_245m.yaml --parameters-only --output-dir reports/scale_245m
```

主配置计数（共享输出权重只计一次）：

| 模式 | Decoder | Reader | Semantic Embedding | Syntax | Composer/Pooling | 总计 |
|---|---:|---:|---:|---:|---:|---:|
| no_memory | 125,793,792 | 0 | 0 | 0 | 0 | 125,793,792 |
| pooling4 | 125,793,792 | 6,297,796 | 5,120,000 | 0 | 198,400 | 137,409,988 |
| fixed_tree4 / random_tree4 | 125,793,792 | 6,297,796 | 5,120,000 | 0 | 656,896 | 137,868,484 |
| forest_scalar4 | 125,793,792 | 6,297,796 | 5,120,000 | 10,861,601 | 656,896 | 148,730,085 |
| forest_vector4 | 125,793,792 | 6,396,100 | 5,120,000 | 10,873,921 | 656,896 | 148,840,709 |

`reports/*/parameters.csv` 提供三种规模的全部模块计数。没有做额外容量匹配；如调整 `syntax_hidden/layers`、`composer_hidden` 或 `reader_width` 等真实参与计算的维度，应保留新 YAML 并报告调整。每次训练自动输出参数 JSON、环境 JSON、真实 Token/s、步骤秒数和峰值分配显存；不会从参数量虚构吞吐量或 GPU 成本。

## 训练、恢复与 Colab 持久化

AdamW：Decoder LR=3e-4，辅助模块 LR=5e-4，betas=(0.9,0.95)，矩阵 Weight Decay=0.1，梯度裁剪 1。默认有效更新 4096 个 CE Token、warmup 200 步、500M Token 余弦终点、最低 10%。第一轮 Seed 42，支持 13/42/2026。

每个 micro-batch 累积 `loss_sum / update_target_tokens`；Padding 不计入分母。更新与检查点边界可切分一句的标签：下次重算完整前缀，已消费标签设为 -100，绝不丢失、重复计数或为凑预算改变样本顺序。因此恢复状态包括 `epoch,cursor,token_offset`，不依赖 DataLoader 预取状态。CPU 回归测试包含 Dropout，并验证恢复后的参数、优化器轨迹、调度、RNG 与数据游标。

检查点保存 Model、Optimizer、Scheduler、Python/NumPy/Torch/CUDA RNG、采样位置、配置、Token 累计、时间累计、最近梯度范数及完整日志行，换目录恢复也能保留曲线。临时文件写完 flush/fsync 后原子替换。本地每 100 步保存 `latest.pt`，关键 Token 边界保存独立文件并同步到配置的 `train.drive_dir`；Drive 使用临时副本再 rename，远程文件系统的最终原子性取决于挂载实现。恢复时选择已完成的最大 Token 检查点；`latest.pt` 可能比最近的里程碑文件旧。

数据指纹、Seed、模型与关键优化/调度配置变化会拒绝恢复。允许提高预算、更换输出/Drive 位置和设备；减小 micro-batch 会保留数据顺序，但 Dropout 与浮点运算的轨迹可能变化，程序会提示。严格 GPU 复现可设 `deterministic: true`，仍应按硬件/精度的数值容差判断。

CUDA 且支持 BF16 时使用 autocast，无需 FP16 GradScaler；CPU 或不支持 BF16 时退回 FP32。OOM 提示减小 `--micro-batch-size`，保持有效 Token 不变，并从最近完整检查点恢复。此实现保留 Python chart 调度，GPU 上的小矩阵与 launch 开销可能限制吞吐量；尚不包含自定义 fused CKY Kernel 或多 GPU。

训练默认显示 tqdm 进度条，以真正参与 CE 的目标 Token 更新，包含当前累积 micro-batch 的进度；恢复时从检查点的累计 Token 开始。每次优化更新显示 step、训练 NLL、验证 NLL 和该更新的 Token/s，验证期间显示 `Validate`。CSV/JSONL 的格式和检查点计数保持一致。无交互日志环境可用 `--no-progress` 或 YAML 的 `train.progress: false` 关闭进度条。

## CUDA 性能剖析

`profile_performance.py` 测量驻留 GPU 的小批次，不下载数据、不保存模型。默认 BF16、预热 2 次、测量 5 次并报告中位数，包括 forward/backward/梯度裁剪/AdamW；每个 micro-batch 单独更新，不含梯度累积、验证、数据搬运或保存。合成输入用于控制历史长度，`--source cache` 可改用真实缓存中的不同故事。

```bash
python profile_performance.py --config configs/smoke.yaml --batch-size 2 --lengths 8 16 24 --warmup 2 --repeats 5 --trace --output-dir runs/cuda_profile_smoke
python profile_performance.py --config configs/main_125m.yaml --batch-size 1 --lengths 24 --trace --output-dir runs/cuda_profile_main
python profile_performance.py --config configs/smoke.yaml --batch-size 2 --lengths 24 --source cache --cache-dir cache/tinystories_integration --output-dir runs/cuda_profile_real
```

输出 `summary.csv`（无 instrumentation 的完整更新）、`components.csv`（单独的分阶段诊断）、算子 CSV/文本、Python cProfile 和可用 Perfetto 打开的 Chrome trace。分阶段诊断仅为计时而临时切断并分阶段回传 autograd 边界，训练源码仍使用单次 backward；这些带同步屏障的时间不能与真实吞吐量计时直接相加。CUDA Event 的区间也包含等待 CPU 提交下一批 Kernel 的空档，不代表纯 GPU 运算时间。Profiler 的嵌套范围/设备范围不能累加成占比，应结合无 profiler 的计时和原生算子事件判断。实际测量与结论见 `reports/CUDA_PERFORMANCE.md`。

多模式、多个 Seed 的单 GPU 顺序实验：

```bash
python experiments.py --config configs/main_125m.yaml --seeds 42 --evaluate
python experiments.py --config configs/main_125m.yaml --seeds 13 42 2026 --output-dir runs/multiseed --evaluate
```

`--dry-run` 查看实际命令。每次运行采用同一缓存、Seed 派生样本顺序、Decoder 和优化器；`matrix_summary.csv` 汇总实测参数、时间、Token/s 与验证 NLL。不同 mode 的随机操作不影响独立 NumPy 采样器的数据顺序。

## 评估与统计

```bash
python evaluate.py --checkpoints runs/main_vector_seed42/tokens_5000000.pt runs/main_vector_seed42/tokens_25000000.pt --cache-dir cache/tinystories --output-dir runs/eval --diagnostics --top1
python evaluate.py --logs runs/main_vector_seed42/metrics.csv --output-dir runs/curves --nll-threshold 2.5
python evaluate.py --bootstrap runs/eval/modelA_stories.csv runs/eval/modelB_stories.csv --output-dir runs/comparison
python semantic_tests.py --cache-dir cache/tinystories --checkpoint runs/main_vector_seed42/tokens_25000000.pt
```

文件名以实际评估生成的 `*_stories.csv` 为准。默认用 dev；选择模型/超参数之后才使用 `--split test`。训练期验证默认最多 100 个 batch，日志曲线和阈值均基于这一固定开发子集；最终 `evaluate.py` 默认评估完整 split。不同曲线必须采用同一个验证范围。

输出：NLL（nats/token）、PPL、前 8 个及后续 Token NLL、Memory Off/Shuffled Memory 差值、每故事 loss sum/Token、语法与 Composer 最近训练梯度范数、结构指标、CSV 汇总及 PNG 曲线。没有后续 Token 时对应 NLL 为 null，没有 CUDA 时 GPU 小时/显存为 null，不填造数据。

Shuffled Memory 在 batch 内选择尽量跨故事的循环置换；只在源故事不同的样本子集上计算配对差，并明确输出该子集的 Token 数。不能跨故事置换的 singleton/同故事样本不计入该项；正式评估宜用足够大的 batch。正差值定义为 `intervention NLL - normal NLL`，表示正常记忆有帮助。吞吐量区分训练 step 时间与评估正常前向时间；打开树导出/诊断会增加成本，比较时保持开关一致。

结构指标定义：根熵为本句保留候选的先验熵；语义余弦在有效无序候选对上平均；Collapse Rate 为存在至少一个余弦>=0.99 同伴的候选比例（排除单候选句）；Attention Entropy 在全部历史/候选槽位上计算，按有效目标、head、Reader 层平均；左右分支比例按所有保留树的内部左/右子边计数。没有 Gold Parse，不报告 Constituency F1。

AULC 用梯形积分，只有覆盖 5M–100M 两端的验证观测才报告；中间线性插值，不外推未训练区间。达到 NLL 阈值使用第一次实际观测，不伪造精确交叉时间。Bootstrap 以故事为单位成对重采样，计算 Token 加权 `NLL(A)-NLL(B)` 的百分位置信区间，检查故事集合、Token 数和缓存指纹。

探索性模板包含 PP Attachment、跨句指代、上下文解释切换与组合重组。每对交换上下文并复用两个续句，保证答案角色的词频完全平衡；生成器按本次 BPE 检查相等续句长度，不合格模板记入 rejected。比较含 EOS 的完整条件 log-likelihood 差，不用长度归一化分数代替要求的差值。它们含人为表面线索，不能排除所有启发式策略；不冒充人工语义标注基准，也不声称组合词组未出现在 TinyStories 训练集。

## 分阶段验证

```bash
# 阶段 1：数据、teacher forcing、普通 LM 学习与因果性
python -m pytest tests/test_data_decoder.py -q
python -m data.prepare --config configs/smoke.yaml --demo
python train.py --config configs/smoke.yaml --mode no_memory --budget-tokens 256 --output-dir runs/stage1
# 阶段 2：精确 CKY、剪枝、非对称性、先验归一化、硬选择限制、探索
python -m pytest tests/test_forest.py -q
# 阶段 3：LM 梯度、Reader、完整因果性、历史边界、空历史
python -m pytest tests/test_model.py -q
# 阶段 4：全部对照组的 forward/backward/step/eval
python -m pytest tests/test_model.py -k all_modes -q
# 阶段 5：缓存、精确预算、完整恢复、评估、Bootstrap、语义模板
python -m pytest tests/test_resume_evaluation.py tests/test_probes.py -q
python -m pytest -q
```

CUDA 测试在没有 CUDA 版 PyTorch 时会明确 skip；Colab 有 GPU 时会实际执行 BF16/SDPA 前向反向。测试临时文件固定放在项目内 `.pytest_tmp`，避免宿主沙箱临时目录权限问题。`reports/VALIDATION.md` 记录本次实际执行范围及限制。

## 文件

`data/prepare.py,dataset.py`：原始文本、Tokenizer、缓存、采样与 Collator；`models/`：Decoder、Syntax、Composer、Forest、Reader、模型接口；`train.py`：优化/计数/保存恢复；`evaluate.py`：指标/统计/绘图；`semantic_tests.py`：模板生成与条件似然；`experiments.py`：矩阵实验和参数统计；`configs/`、`tests/`、Notebook、本 README。


## CKY V2.5-P precision study

`configs/precision/{bf16,fp32_ieee,fp32_tf32,fp16}.yaml` 为 BF16 外层 Transformer 配置独立 Composer 精度。参数与 FP32 IEEE/TF32 树状态保持 FP32；FP16 组显式保存 Half 状态并使用 GradScaler，checkpoint 保存策略和 scaler，恢复时严格核对。原 BF16 AMP Composer 的 LayerNorm 输出其实为 FP32，详见真实 dtype trace。

```powershell
& 'E:/anaconda3/envs/py312/python.exe' run_precision_study.py --resume
# 单独实验
& 'E:/anaconda3/envs/py312/python.exe' validate_precision_v25.py --resume
& 'E:/anaconda3/envs/py312/python.exe' benchmark_precision_v25.py --resume
& 'E:/anaconda3/envs/py312/python.exe' train_precision_stability.py --resume
& 'E:/anaconda3/envs/py312/python.exe' train.py --config configs/precision/fp32_ieee.yaml --forest-implementation lazy_semantic --output-dir runs/fp32_composer
```

每个后端精度策略独立子进程，TF32 仅使用新版 `fp32_precision` API，要求支持该 API 的 CUDA PyTorch（本机 2.12.1+cu132；精度依赖见 `requirements-precision.txt`）。默认仍为 V2，未进入 V3。研究脚本固定 Score 原路径，另设 score FP32 控制，不混同归因。训练稳定性使用用户选择的 smoke 模型、104 个未再截断的真实缓存样本、effective CE tokens=16，完整固定 dev=1041 CE tokens；这些局部实验不代表大规模预训练结论。

[完整精度报告](reports/cky_v25_precision/summary.md) 与原始 CSV/JSON 保留严格数值失败。`PrecisionStudy_Colab.ipynb` 和 `precision_study_cache.zip` 提供相同输入的第二种 GPU 入口；更换 GPU 必须使用新的 `--output-dir`，不能混入本机数据。当前没有实际 Blackwell 测量。


## CKY V2.5-R Reader precision isolation

`precision.reader: fp32_ieee` 可独立将 Semantic Reader 放入禁用 AMP 的 FP32 IEEE 区域，保留可微 cast、原生 SDPA、Dropout 和原公式；返回 dtype 维持 Decoder 接口。默认 `reader: bf16` 继承外层 AMP，默认 CKY 仍为 `score_first`。旧 precision checkpoint 缺失该字段时按原默认兼容。

```powershell
& 'E:/anaconda3/envs/py312/python.exe' validate_reader_precision.py
& 'E:/anaconda3/envs/py312/python.exe' benchmark_reader_precision.py --repeats 8
& 'E:/anaconda3/envs/py312/python.exe' audit_reader_search.py --batch 1
& 'E:/anaconda3/envs/py312/python.exe' audit_reader_search.py --batch 4
& 'E:/anaconda3/envs/py312/python.exe' audit_reader_profiler.py
& 'E:/anaconda3/envs/py312/python.exe' summarize_reader_precision.py
# 可选 Reader FP32 的训练配置入口；本轮没有执行训练
# train.py --config configs/precision/fp32_reader.yaml --forest-implementation lazy_semantic
```

数值实验主模型 batch1/4、W24/H4/K4/R12，B/R/E 六组配对均完成；五种性能形状同轮四臂，各 warmup2+采样8。FP32 Reader 大幅减小投影误差，但完整梯度相对 L2 仅降低9.39%/6.67%，BF16 严格梯度容差仍未通过。建议保留 BF16 Reader，FP32 作为实验选项。未做局部 KV/Attention、额外训练或 V3。详情见 [九节实验报告](reports/cky_v25_reader/summary.md)、[原始性能](reports/cky_v25_reader/benchmark.csv)、[完整性审计](reports/cky_v25_reader/completion.json)。脚本顺序运行，GPU benchmark 期间不要并发其他 GPU 实验。

## CKY V3 Width-batched Tensorization

`lazy_tensorized` 使用按 width 的可微分数张量、缓存 ragged Gather/候选 ID 和根装配，保留原 Top-R/Top-K/两级探索、BF16 叶子加法顺序及同一 Lazy DAG/Composer 尾部。`lazy_tensorized_a` 保留 A 阶段 Cell/root 适配，用于阶段对照。默认仍为 `score_first`，可用配置 `configs/v3.yaml`（smoke）、`configs/v3_main_125m.yaml`、`configs/v3_fp32.yaml` 显式选择。无界 K=None/R=None 精确枚举使用旧搜索兼容路径，失败日志及严格测试保留。

```powershell
& 'E:/anaconda3/envs/py312/python.exe' validate_cky_v3.py
& 'E:/anaconda3/envs/py312/python.exe' train_cky_v3_validation.py
& 'E:/anaconda3/envs/py312/python.exe' benchmark_cky_v3.py
& 'E:/anaconda3/envs/py312/python.exe' benchmark_cky_v3.py --precision fp32 --cases main_b1_w24 main_b4_w24 --resume
& 'E:/anaconda3/envs/py312/python.exe' profile_cky_v3_widths.py
& 'E:/anaconda3/envs/py312/python.exe' summarize_cky_v3.py
```

已完成 1080 结构矩阵、36 边界、10 组完整梯度配对；413 tests passed、1 skipped。CPU FP32 8 步、真实 CUDA 从旧 Lazy 500 步续至 520 步及 V3 checkpoint 重放通过；另有 microbatch1 的真实梯度累积吞吐。九组最终配对（七种 BF16 形状、两个 IEEE 主模型）各侧 warmup2+有效8样本，另有 A/Lazy 与 B/A 的独立同轮测量。主模型 BF16 B4 update=545.6→297.1ms，配对中位倍率1.831x；完整 launches=14908→6332。独立缓存峰值2.737→2.741GiB，Composer785行/23次保持。只实测 RTX3070 Laptop，未进行编译、自定义 kernel 或 Blackwell 外推。

[十节完整报告](reports/cky_v3/summary.md)、[性能 CSV](reports/cky_v3/benchmark.csv)、[完整性审计](reports/cky_v3/completion.json) 包含组件/逐 width/profile/显存口径与失败回退。CUDA恢复脚本需要已有可信 Lazy checkpoint（默认本机 V2.5-P B2 seed42），大权重不打入源码包。更换 GPU 或参数应使用新的 output-dir；测速与其他 GPU 任务顺序运行。早期预训练仍需受控验证，短程轨迹不证明质量等价。
