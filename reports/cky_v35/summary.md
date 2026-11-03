# CKY V3.5：Performance Ceiling Investigation

本轮不改变任何模型数学、搜索预算（K=4，R=12，Delta=8，exploration=0.1）、Loss、数据、Tokenizer、参数或 state_dict。正式训练默认实现仍为 `score_first`；所有新实现均通过配置切换（`configs/v35/`）。文中所有数字均来自本轮在本机重新测量，不与 V3 报告或其它 GPU 的数字相除。

## 0. 环境与前提

| 项目 | 值 |
|---|---|
| GPU | NVIDIA GeForce RTX 3070 Laptop GPU，8.00 GiB，sm_86，40 SM，驱动 617.42 |
| PyTorch / CUDA | 2.12.1+cu132 / 13.2，Python 3.12.13（`E:/anaconda3/envs/py312`） |
| Triton | 本轮安装 `triton-windows 3.8.0.post29`（inductor 在 Windows 上需要它；见 `requirements-v35.txt`） |
| 主模型 | `configs/v3_main_125m.yaml`：16 层 Decoder、hidden 768、H=4、W=24、K=4、R=12，BF16 outer AMP、FP32 IEEE Composer、BF16 Reader、score_dtype existing，148.8M 参数 |
| 权重 | seed 42 初始化。本机 `runs/main_vector_seed42/{latest,tokens_5000000}.pt` 均为截断的 zip（130 MB，149M 参数的完整 checkpoint 应约 1.8 GB），无法加载；因此主模型没有已训练权重可用，与 V3 协议一致使用固定初始化 |
| 数据 | `cache/tinystories` 完整缓存（train 4074 万句、4.83 亿目标 token），fingerprint `2bf7443a…bf28a30`；性能探针使用独立 sampler seed 7350001（非训练 seed 13/42/2026，非 dev split） |

**本机显存限制（影响解读）**：这台笔记本的 GPU 同时被桌面进程占用 2–4 GB（Chrome、WeChat 等，随时间波动）。Windows WDDM 下 PyTorch 不会立刻 OOM，而是把分配溢出到共享系统内存，吞吐骤降 5–30 倍且极不稳定。实测主模型 micro-batch ≥16 多次落入这种状态（见 §1.2）。因此本机**可靠**的主模型数据只有 micro-batch ≤8（峰值 ≤3.2 GiB）；≥16 的数字只能作为下界参考，目标 GPU（RTX PRO 6000 Blackwell）必须重新测量。重跑的 part 2 使用 `torch.cuda.set_per_process_memory_fraction` 把分配器封顶在启动时的空闲显存，使不适配的配置以 OOM 记录而非悄悄溢出。

## 1. Stage A：真实训练性能审计

脚本：`profile_cky_v35.py`；输出：`reports/cky_v35/audit/`；表格：`reports/cky_v35/tables/audit_tables.md`。

三类时间必须分开读：**Host wall**（CPU 发射/阻塞时间）、**CUDA event envelope**（GPU 时间线上该段的起止跨度，包含发射空档与等待）、**Kernel active**（profiler 中 kernel 实际执行时间，来自单独的 profiled 运行，带 profiler 开销）。它们不能相加成精确分解。

### 1.1 真实数据 micro-batch sweep（固定 effective_target_tokens=4096）

复用 `train.train_update` 的语义（SUM loss / 精确 token 预算 / clip / AdamW / scheduler），含 mmap 取样、Collator、H2D；排除验证与 checkpoint IO。每个 micro-batch 大小看到相同的样本序列，只是切分不同。

| micro-batch | 状态 | 每 update 中位 ms | 最小 ms | micro-batches/update | 每 micro-batch ms | CE token/s | 峰值 GiB |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | ok | 67076 | 64491 | 346 | 194 | 61.1 | 2.72 |
| 2 | ok | 35854 | 35341 | 173 | 207 | 114.5 | 2.73 |
| 4 | ok | 23816 | 19581 | 87 | 274 | 182.7 | 2.76 |
| 8 | ok | 16901 | 15914 | 44 | 384 | 187.7 | 3.17 |
| 16 | 不可靠（共享内存溢出） | 104755 | 47698 | 22 | — | 20.5 | 3.83 |
| 32 | 不可靠（共享内存溢出） | 159738 | 55017 | 11 | — | 15.3 | 5.16 |
| 64 | 2 次成功后 OOM | 10163 | 7256 | 6 | — | 403（仅 2 样本） | 6.51 |
| 128 | OOM | — | — | — | — | — | 6.54 |

- micro-batch 1→8：每个 micro-batch 的时间从 194 ms 涨到 384 ms，而 CE token 从 11.9 涨到 94（8×），吞吐 61→188 token/s。mbs 8 的单 micro-batch 时间比 mbs 4 多 110 ms，主要来自 Composer 行数（DAG 节点数随行数线性增长）与 backward 节点增多。
- mbs 64 在显存恰好空闲的两次 update 中达到 7.3–13.1 s/4096 token（≈400 token/s），说明在显存充足的 GPU 上吞吐随 micro-batch 继续上升；这是 2 个样本的下界参考，不是本机结论。
- 每 update 的采样+Collation 仅 83–129 ms（<0.5%），H2D <0.3%。数据管线不是瓶颈。

Stage envelope 占 update 墙钟的比例（mbs 1/2/4/8）：forward 50–51%，其中 forest 30–36%、decoder 10–16%、reader 4–6%、syntax 3–4%；backward 47–50%；clip+AdamW 0.3–0.6%（每 update 只执行一次，但每 micro-batch 时它占 10%）。

### 1.2 单个真实 micro-batch 的分解（mbs 4 / 8 / 16，各 3 个真实 batch 的中位数）

| 量 | mbs 4 | mbs 8 | mbs 16 |
|---|---:|---:|---:|
| 未加仪器的 step wall ms（fwd+bwd+clip+AdamW，输入驻留） | 334.9 | 326.6 | 347.5 |
| CE tokens | 30 | 75 | 168 |
| GPU kernel busy（并集，profiled）ms | 60.8 | 62.2 | 79.5 |
| GPU 空闲比例（相对 profiled envelope） | 0.88 | 0.86 | 0.84 |
| cudaLaunchKernel 次数 | 5000 | 5026 | 5675 |
| forward envelope ms | 157.3 | 146.4 | 168.0 |
| 　syntax encoder（含 span scorer） | 9.4 | 9.5 | 9.2 |
| 　forest 总计 | 103.0 | 97.2 | 112.4 |
| 　　score-only CKY search | 69.4 | 58.9 | 68.3 |
| 　　D2H 阻塞 / CPU DAG 闭包 / H2D plan | 1.0 / 1.9 / 0.4 | 2.0 / 3.2 / 0.5 | 3.7 / 5.5 / 0.5 |
| 　　Composer（width-batched） | 17.5 | 17.3 | 18.1 |
| 　readers（4 层合计） | 15.4 | 13.9 | 12.4 |
| 　decoder（含 readers） | 41.4 | 38.1 | 34.0 |
| backward envelope ms | 144.1 | 130.7 | 131.5 |
| 　staged：CKY backward（含 Composer backward） | 91.2 | 84.1 | 88.2 |
| 　　其中 Composer backward | 37.4 | 33.7 | 36.2 |
| 　staged：decoder+reader backward | 59.7 | 54.3 | 54.8 |
| 　staged：syntax+embedding backward | 14.4 | 12.5 | 12.6 |
| clip / AdamW envelope ms | 7.5 / 32.9 | 7.6 / 30.9 | 7.0 / 29.2 |
| kernel active：search 全部 width | 2.9 | 3.0 | 4.3 |
| kernel active：Composer fwd / bwd | 0.9 / 1.9 | 0.9 / 1.9 | 1.2 / 2.6 |
| kernel active：decoder fwd / bwd | 3.6 / 6.1 | 4.1 / 7.4 | 6.8 / 12.5 |
| kernel active：CKY backward（staged） | 2.2 | 2.2 | 3.5 |
| kernel active：AdamW / clip | 32.4 / 4.3 | 30.9 / 4.3 | 29.1 / 4.4 |
| launches：search / Composer / decoder / readers / backward / AdamW | 1147 / 244 / 376 / 244 / 2575 / 30 | 1147 / 255 / 377 / 236 / 2595 / 30 | 1390 / 314 / 329 / 236 / 2923 / 30 |
| score→log-prior autograd 节点（含 Syntax） | 679 | 679 | 781 |
| loss 全图 autograd 节点 | 2346 | 2346 | 2526 |
| cProfile：项目 Python / torch Python / 内建(C 扩展调用) ms | 40.5 / 15.6 / 316.0 | 32.4 / 12.4 / 246.5 | 44.1 / 14.7 / 303.2 |

**Q1 结论（时间到底花在哪里）**：

1. 一个真实 micro-batch 约 330 ms，GPU kernel 实际忙 60–80 ms，**84–88% 的时间 GPU 空闲**。5000–5700 次 kernel launch，平均每次 launch 全程（Python 调用、dispatcher、autograd 记录、驱动提交）约 55–65 µs，这就是步长。
2. 按 envelope 排序：CKY backward 84–91 ms（kernel 仅 2–4 ms），score search 59–69 ms（kernel 仅 3–4 ms，1150–1390 次 launch），decoder+reader backward 55–60 ms，decoder forward 34–41 ms，Composer fwd+bwd 51–55 ms（kernel 3–4 ms），AdamW 29–33 ms（kernel 29–32 ms，是**唯一**被 GPU 真正填满的阶段，占全部 kernel 时间的 40–50%），syntax fwd+bwd ≈ 24 ms，clip 7 ms。
3. CPU/GPU 同步：D2H 之前 CPU 实际等待 GPU 仅 0.02–0.06 ms（GPU 早已算完，CPU 才是落后的一方），拷贝+tolist 1–3 ms，CPU DAG 闭包 1.9–5.5 ms，plan H2D 0.4–0.5 ms。**Lazy DAG 的 CPU/GPU 同步不是瓶颈**（<3% 的 step）。每 step 有 152 次 `cudaMemcpyAsync`（小索引张量 H2D 与标量读取，CPU 侧合计 6.6 ms）。
4. 数据读取/Collation 与 H2D 合计 <1%。
5. Python 项目代码的独占时间只有 32–44 ms（10–13%）；316 ms 的"内建"时间是 torch C++ 算子调用本身——瓶颈在算子数量，不在 Python 逻辑。
6. 随 micro-batch 从 4 到 16，step wall 几乎不变（335→348 ms）而 kernel busy 从 61→79 ms：launch 数几乎与 batch 无关，GPU 工作量才与 batch 成正比。

### 1.3 真实长度分布、Padding 与候选利用率（100 个 update 的探针流）

| micro-batch | micro-batches/update | 平均 padded words | 平均 history slots | word padding 比例 | span 利用率 | target padding | 重复历史句比例 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 344.7 | 13.4 | 4.0 | 0.274 | 0.615 | 0.000 | 0.0002 |
| 2 | 172.6 | 16.2 | 4.0 | 0.424 | 0.438 | 0.187 | 0.0003 |
| 4 | 86.5 | 18.4 | 4.0 | 0.499 | 0.345 | 0.271 | 0.0005 |
| 8 | 43.6 | 20.4 | 4.0 | 0.547 | 0.286 | 0.337 | 0.0008 |
| 16 | 22.0 | 22.0 | 4.0 | 0.582 | 0.245 | 0.388 | 0.0015 |
| 32 | 11.2 | 23.1 | 4.0 | 0.603 | 0.222 | 0.422 | 0.0025 |
| 64 | 6.0 | 23.7 | 4.0 | 0.613 | 0.210 | 0.457 | 0.0044 |
| 128 | 3.0 | 24.0 | 4.0 | 0.620 | 0.203 | 0.487 | 0.0069 |

- 句长均值 10.8 词（p50 10，p95 19，最大 24），目标长度均值 11.9 BPE。mbs≥8 时几乎每个 batch 都被最长句撑到 22–24 个词，**chart 中 71–80% 的 span 属于 padding**。
- 但 §1.2 显示 search 的 GPU kernel 时间只有 3–4 ms；padding 浪费的是 GPU 工作量（本来就只占 1%），不是时间。只有当 width 数（= padded words −1）减少、launch 数随之减少时，长度分桶才能变成时间收益：search launch ≈ 50/width，CKY backward 与 Composer 的 launch 也按 width 计。粗略估计把 24 词桶换成 13 词桶可省 ≈11 个 width × (50+~45+~25) ≈ 1300 次 launch ≈ 70–80 ms/step（约 22%），**这是未验证的估计**（见 §5）。
- 同一 micro-batch 内完整相同（token 序列+有效长度）的历史句比例 ≤0.7%：**历史句去重不值得实现**。

### 1.4 静态索引缓存（LRU 8）

缓存 key = (padded words, B×padded history slots)。真实数据中 padded words 随 batch 变化（mbs 1 时 84 个不同 key，mbs 4 时 57 个），大 batch 时每个 update 末尾的"余数" micro-batch 行数各不相同（如 rows=356/364/368…），也产生新 key。

| micro-batch | 探针流 LRU8 命中率 | LRU16 | LRU32 | LRU64 | 计时 update 中实测命中率 | 构建 ms/update | 构建占 update |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.336 | 0.612 | 0.881 | 0.987 | 0.323 | 3645 | 5.4% |
| 2 | 0.510 | 0.867 | 0.970 | 0.994 | 0.490 | 1873 | 5.2% |
| 4 | 0.652 | 0.983 | 0.991 | 0.993 | 0.662 | 948 | 4.2% |
| 8 | 0.780 | 0.975 | 0.982 | 0.985 | 0.795 | 1150 | 5.3% |
| 16 | 0.885 | 0.954 | 0.960 | 0.964 | — | — | — |
| 32 | 0.879 | 0.908 | 0.921 | 0.930 | — | — | — |
| 64 | 0.807 | 0.838 | 0.857 | 0.880 | — | — | — |
| 128 | 0.730 | 0.780 | 0.873 | 0.877 | — | — | — |

- 原构建器（Python 逐行列表）冷启动成本：w=24 时 n=4 需 31 ms，n=16 需 38 ms，n=64 需 49–56 ms，n=128 需 80–111 ms，n=512 需 174–187 ms（`cache_analysis.json` 的 builds 表，共 290 个实测 key）。
- 真实训练中缓存构建占 update 墙钟 **4–5%**（mbs 1–8），比 V3 报告基于固定 shape 的"预热后 100% 命中"假设差得多。被驱逐最多的是中间长度的 key（如 w=13–17），因为相邻 batch 的 padded words 在 11–24 之间跳动。
- 容量提高到 32 可把 mbs ≤8 的命中率提到 97–99%；但 mbs≥32 的余数 batch 产生的新 key 与容量无关，需要降低单次构建成本。两者都在 Stage D 处理，不改变任何训练样本。

### 1.5 不必要计算清单（Q2）

| 现象 | 证据 | 可消除？ |
|---|---|---|
| 搜索阶段为全部 width 保留可微中间分数（index_select/cat/gather 反向节点） | score→prior 679–781 个 autograd 节点；staged CKY backward 84–91 ms 中 kernel 仅 2–4 ms | 是（V3-C，§2） |
| 每 width ~50 次 search launch + ~45 次反向 launch（gather/mask/topk/rand 周边小 kernel） | width 子阶段表：selection 522–726 次、gather 253–412 次、pair 171–231 次、assembly 167–227 次 | 部分（compile 融合，§3） |
| 结构索引重复构建 | 缓存命中率 32–80%，每 update 4–5% 时间 | 是（Stage D） |
| Padding 候选（71–80% 的 span 无效） | §1.3 | GPU 工作量上可消除，但当前不是时间瓶颈 |
| 重复历史句 | ≤0.7% | 不值得 |
| CPU 读 GPU 导致停顿 | D2H 前 GPU 等待 ≈0 | 不是问题 |
| AdamW foreach kernels 占 GPU kernel 时间 40–50% | top kernels 全部是 Foreach* | 与 CKY 无关；fused AdamW 是单独的工程选项（Stage D 仅做诊断测量，未改默认） |

## 2. Stage B：V3-C Score Reconstruction

实现：`models/forest_score_reconstruct.py`（实现名 `lazy_tensorized_score_reconstruct`），`_finish_lazy` 增加一个可选 `reconstruct` 回调（`models/forest_lazy.py`），V3 的审计尾部提取为 `_record_tensorized_audit`。不新增参数、不改 state_dict；checkpoint 记录 `forest_implementation` / `forest_search_backend` / `forest_reconstruction_mode` 与 precision policy。

### 2.1 数学与实现

- V3 搜索在 `torch.no_grad()` 下原样执行（同一函数、同一 RNG 消耗顺序、同一 Top-R/Top-K/探索/Delta），只产出候选 ID、backpointer、mask 与无梯度分数。
- Lazy tail 的 CPU 闭包给出按 width 分层的紧凑 DAG plan（left/right/parent 索引），原本只用于 Composer 与 vector coefficient。重建沿同一 plan 计算  
  `ℓ_P = (bank[li] + bank[ri]).float() + s_flat[ui]`，  
  其中 bank 的第一层是 `span_scores.diagonal()` 经 leaf mask 置零后的叶子分数（与 V3 bank 相同的张量与 dtype），之后每层 `torch.cat` 累积。width 2 的两个叶子在 BF16 下仍先以 BF16 相加再 `.float()`；width ≥3 的 bank 因 cat 提升为 FP32，与 V3 的 `(ls+rs).float()+local` 逐位相同。`local` 使用 `span_scores.float()` 的 gather：前向数值与 FP32+BF16 的类型提升完全一致，反向则像 V3 一样先在 FP32 里累加同一 span 的全部候选贡献、再一次性 cast 回 BF16。
- 根分数 = 节点 bank 按 root plan gather，`w>1` 时 cast FP32（V3 root bank 的 dtype），无效槽置 0；`log_prior=masked_log_softmax(root_s,root_m)` 不变。
- 审查结论：只有根分数进入 LM Loss（无中间分数辅助损失）；探索只依赖 `score.detach()` 与 rand；vector coefficient 走原 Composer plan，不在分数重建里；同一棵树中一个 span 只出现一次，不同根树共享节点时梯度通过 index_select 反向累加；共同 span bias 方向在两种实现中都是同一组加法的梯度；BF16 叶子→FP32 的转换时机与 V3 相同（width 2 之后）。
- 三种模式：`dag`（C1，共享节点各算一次）、`tree`（C2，按根树展开占位、不共享，仅作参考）、`linear`（C3，根分数 = 树内全部 span 势的一次 FP32 求和；**改变浮点加法顺序，前向不逐位相等，仅作实验，配置不选用**）。
- K=None 且 R=None 的精确枚举保持 V3 的旧可微搜索（与 V3 相同的兼容分支）。

### 2.2 正确性（`validate_cky_v35.py`，`tests/test_score_reconstruction.py`）

- 搜索身份矩阵：B∈{1,2,4}×W∈{1,4,8,16,24}×K∈{1,2,4}×R∈{4,12}×{scalar,vector}×{normal,explore}，CPU FP32 / CUDA FP32 / CUDA BF16，共 **1080 对 + 36 个边界用例（ties/near_ties/prune/empty/all_inf/partial_inf）全部通过**：每个 span 的 selected/retained ID、mask、分数逐位相同，四种 RNG 状态相同，根 semantic/scores/coeff/log_prior 逐位相同，trees 相同。
- 完整 LM 梯度（旧严格容差 FP32 atol 3e-6/rtol 3e-4，BF16 atol 1.5e-4/rtol 0.04，与 lazy_semantic 参考比较，与 V3 验证同协议）：

| device | precision | 模型 | batch | 全局 max abs | 严格失败参数 | logits 逐位相等 | span scorer 梯度范数 |
|---|---|---|---:|---:|---:|---|---:|
| cpu | fp32 | small scalar4 / vector4 | 2 | 3.7e-9 / 1.9e-9 | 0 / 0 | 是 | 0.166 / 0.091 |
| cuda | fp32 | small scalar4 / vector4 | 2 | 1.9e-9 / 1.9e-9 | 0 / 0 | 是 | 0.101 / 0.146 |
| cuda | bf16 | small scalar4 / vector4 | 2 | 6.1e-5 / 1.2e-4 | 0 / 0 | 是 | 0.076 / 0.156 |
| cuda | fp32 | main | 1 / 4 | 4.7e-10 / 4.7e-10 | 0 / 0 | 是 | 0.097 / 0.075 |
| cuda | bf16 | main | 1 / 4 | 2.9e-11 / 1.5e-5 | 0 / 0 | 是 | 0.123 / 0.134 |

- V3 ↔ V3-C 直接配对（同权重同 RNG）：CPU/CUDA FP32/BF16 小模型与 CUDA 主模型 B1/B4：logits 与 loss 逐位相等，严格失败 0，全局梯度 max abs ≤ 7.6e-6（BF16 B4）。
- C1 vs C2：15 个用例前向逐位相同；FP32（CPU/CUDA）梯度全部在严格容差内；CUDA BF16 有 3 个用例的 **BF16 输入张量 `scores` 的梯度**出现 1 个 BF16 ulp 的差异（max abs 0.0039–0.0078，normalized L2 ≤1.2e-4，参数梯度无失败）：C2 把共享子树的占位分别回传，FP32 index_add 的求和分组不同，再 cast 到 BF16 时偶尔翻转一个 ulp。记录为 C2 的已知差异，C1（dag）才是使用的实现。
- 单元测试另验证：搜索输出无 autograd 图；BF16 width-2 叶子加法保持；共享子树梯度累加（两棵根树共用 span 时该 span 势梯度为 2）；精确枚举回退；state_dict 键不变。

### 2.3 Benchmark

见 §4。

## 3. Stage C：局部 torch.compile

实现：`models/cky_compile.py` 把 V3 width 循环体（gather/pair/mask/Top-R/探索/Top-K/Delta/分数 gather/backpointer 解码）提取为纯张量函数 `width_step`，forest 属性 `search_backend ∈ {eager, function, compile_default, compile_reduce_overhead}`；eager 路径的内联循环逐字不变。编译设置（缺一不可）：`fallback_random=True`（rand_like 使用 eager Philox kernel，否则 inductor 自己的 RNG 与 eager 流不同）、`emulate_precision_casts=True`、`recompile_limit=512`（最终版 1024；23 个 width × 多种 shape 远超默认 8），`fullgraph=True`；形状策略（`dynamic`）见下文两次迭代。

**负对照**：不开 `emulate_precision_casts` 时，inductor 把 width-2 的 BF16 叶子加法在 FP32 中计算，W=8 与 W=24 的 CUDA BF16 搜索在 span (0,2)/(1,3)/(2,4) 等 width-2 分数上出现 2.7e-4–4.9e-4 的差异（`compile_identity.json`），这正是上一轮"提前 cast 叶子分数会改变 Top-K 身份"的同类问题；开启后逐位一致。

**形状策略与 CUDA Graph（两次迭代，均记录）**：

1. *第一版（§4 的 paired benchmark 与 §4.6 的流测量所用）*：`dynamic=False`，每个 (rows, words, width) 一张静态图；`reduce-overhead` 关闭 cudagraph trees（非 tree 的每图一张静态缓冲）。关闭 trees 的原因是同一进程里先以 V3（带梯度）、再以 V3-C（`no_grad`）调用同一个编译函数时，trees 把前一代输出判为失效，在 width 循环的 `torch.cat(bank)` 处抛出 "accessing tensor output of CUDAGraphs that has been overwritten by a subsequent run"（首次 benchmark 因此整个 case 失败，日志保留在 `reports/cky_v35/logs/`）。这一版在固定形状的 benchmark 上全部身份验证通过，但在真实训练里失败：`train_update` 为精确命中 token 预算会截断每个 update 的最后一个 micro-batch，行数每个 update 都变，第 6 个 update 就超过 Dynamo 的 `recompile_limit`（§6、失败记录）。
2. *第二版（最终，§6 训练与 §4.7 复测所用）*：把 bank/mask_bank/span_flat/left_ids/right_ids 的行维用 `torch._dynamo.mark_dynamic` 标为符号维（`dynamic=None`，`automatic_dynamic_shapes=False`，其余维度仍静态），每个 (words, width) 只编一张图，行数变化不再重编（`tests/test_cky_compile.py::test_dynamic_rows_do_not_recompile_and_match_eager`）。width step 的运算逐行独立，RNG 消耗顺序不变。非 tree 的 CUDA graph 要求全静态形状，会把符号行维重新特化为常数（`ConstraintViolationError`），因此 `reduce-overhead` 改回 PyTorch 默认的 cudagraph trees，并在每次搜索 forward 开头调用 `torch.compiler.cudagraph_mark_step_begin()`（官方给出的修复），带梯度/`no_grad` 交替的进程内复用随之通过。trees 对每个新的行数重新录制一组图（计数器 `cudagraph_recorded_non_static_inputs`），共享同一内存池。

这两次迭代共同说明：**Lazy DAG 的 CPU 闭包与 D2H 使整个 forward 无法进入一张 CUDA graph，CUDA graph 只能按 width 逐张捕获**，每个 width 仍有一次 graph launch 加 eager 的 cat/bands 开销。

正确性（`compile_identity.json`，第二版配置重跑；第一版的同一验证也全部通过）：`v3_function`、`v3_compile_default`、`v3_compile_ro`、`v3c_compile_default`、`v3c_compile_ro` 各 10 个固定 shape 森林用例（FP32/BF16，normal/explore/ties/near_ties）与 3 个完整 LM 用例（小模型 FP32/BF16，主模型 BF16 B4）全部通过：候选 ID、backpointer、RNG、根分数、log-prior、logits 逐位相等，梯度严格容差内，**graph break 0**。

性能见 §4。

## 4. 端到端 Benchmark（`benchmark_cky_v35.py`）

协议：每个 case 一个进程；六个 arm 共用同一份权重/优化器/RNG 快照，每次采样前恢复；arm 顺序每轮轮换（paired），warmup 3 次、计时 10 次；每步 = 完整 micro-batch update（forward、backward、clip、AdamW，`torch.cuda.synchronize` 前后），不排除任何东西。编译 arm 的首次调用（含编译）单独计时并记录在表内，不计入 steady-state 中位数但也不隐藏。case 命名：`smoke_*` 小模型合成、`main_*` 主模型合成（W=24 满长，B=histories）、`real_b2` 小模型真实 batch、`real_main_mb4/mb16` 主模型真实 micro-batch（mb16 的峰值显存 2.79 GiB，未触发共享内存溢出，因此这里的 mb16 数字可靠；它与 §1 sweep 里的 mb16 不同——sweep 是整轮 4096-token update，多个 batch 叠加分配）。原始样本：`reports/cky_v35/benchmark/performance/bf16/<case>/`；汇总：`reports/cky_v35/benchmark.csv`（含 paired ratio）、`tables/benchmark_tables.md`、`figures/benchmark_arms.png`。

### 4.1 正确性（每个 case、每个 arm 对比同进程的 V3 eager）

48 行中全部 40 个非参考 arm：候选 ID / backpointer / mask 身份 **True**、四种 RNG 状态相同、logits 与 loss **逐位相等**、根分数与 log-prior max abs **0**、全局梯度严格容差失败 **0**（max abs ≤ 7.6e-6，BF16）。V3-C 与编译在所有 benchmark 输入上都没有改变搜索或数值。

### 4.2 完整 micro-batch 步时间（中位数，ms）

| case | v3 | v3c | v3_compile_default | v3_compile_ro | v3c_compile_default | v3c_compile_ro |
|---|---:|---:|---:|---:|---:|---:|
| smoke_b2_w24 | 217.9 | 195.6 | 186.2 | 185.0 | 179.0 | 171.0 |
| smoke_b8_w24 | 216.7 | 216.6 | 205.7 | 179.0 | 191.0 | 174.8 |
| main_b1_w24 | 309.8 | 271.3 | 259.1 | 238.8 | 263.6 | 242.9 |
| main_b4_w24 | 289.2 | 287.7 | 274.3 | 251.1 | 269.3 | 261.6 |
| main_b8_w24 | 297.4 | 299.1 | 278.7 | 249.8 | 274.7 | 261.8 |
| real_b2 | 203.2 | 167.1 | 152.9 | 137.6 | 160.2 | 141.4 |
| real_main_mb4 | 244.1 | 238.2 | 217.3 | 213.6 | 224.7 | 211.9 |
| real_main_mb16 | 269.3 | 261.9 | 247.1 | 241.1 | 260.1 | 245.8 |

### 4.3 Paired 加速比（V3 eager 时间 / arm 时间，同一轮次配对后取中位数；>1 为更快）

| case | v3 | v3c | v3_compile_default | v3_compile_ro | v3c_compile_default | v3c_compile_ro |
|---|---:|---:|---:|---:|---:|---:|
| smoke_b2_w24 | 1.000 | 1.060 | 1.153 | 1.181 | 1.169 | 1.203 |
| smoke_b8_w24 | 1.000 | 1.019 | 1.066 | 1.210 | 1.155 | 1.236 |
| main_b1_w24 | 1.000 | 1.095 | 1.176 | 1.237 | 1.083 | 1.285 |
| main_b4_w24 | 1.000 | 1.029 | 1.064 | 1.163 | 1.076 | 1.118 |
| main_b8_w24 | 1.000 | 0.983 | 1.067 | 1.202 | 1.063 | 1.153 |
| real_b2 | 1.000 | 1.194 | 1.220 | 1.461 | 1.284 | 1.372 |
| real_main_mb4 | 1.000 | 1.006 | 1.086 | 1.157 | 1.041 | 1.149 |
| real_main_mb16 | 1.000 | 1.016 | 1.075 | 1.128 | 1.106 | 1.159 |

paired 比值的最小值/最大值见 `tables/benchmark_tables.md`：所有 arm 的 min 都 <1（单轮抖动 ±10–30%，这台机器的 CPU 调度噪声很大），因此只有中位数 ≥1.10 的差异才算稳健。

### 4.4 为什么快／为什么不快

每步 CUDA launch 数（profiler 统计，含 cudaGraphLaunch）：

| case | v3 | v3c | v3_compile_default | v3_compile_ro | v3c_compile_default | v3c_compile_ro |
|---|---:|---:|---:|---:|---:|---:|
| smoke_b2_w24 | 5309 | 5280 | 4053 | 3888 | 4344 | 4204 |
| smoke_b8_w24 | 5341 | 5312 | 4085 | 3920 | 4376 | 4236 |
| main_b1_w24 | 6276 | 6247 | 5020 | 4855 | 5311 | 5171 |
| main_b4_w24 | 6332 | 6303 | 5076 | 4911 | 5367 | 5227 |
| main_b8_w24 | 6301 | 6272 | 5045 | 4880 | 5336 | 5196 |
| real_b2 | 4229 | 4174 | 3189 | 3064 | 3398 | 3294 |
| real_main_mb4 | 4999 | 4982 | 4073 | 3962 | 4292 | 4200 |
| real_main_mb16 | 5674 | 5651 | 4577 | 4445 | 4832 | 4722 |

Score search 前向 host wall（ms，forward 内从 `_search_scores` 进入到返回）：

| case | v3 | v3c | v3_compile_default | v3_compile_ro | v3c_compile_default | v3c_compile_ro |
|---|---:|---:|---:|---:|---:|---:|
| smoke_b2_w24 | 66.7 | 66.2 | 44.4 | 36.4 | 41.5 | 38.8 |
| smoke_b8_w24 | 64.9 | 86.7 | 47.1 | 36.2 | 41.7 | 55.7 |
| main_b1_w24 | 61.6 | 66.6 | 65.3 | 41.9 | 62.0 | 46.3 |
| main_b4_w24 | 70.8 | 96.5 | 45.4 | 38.6 | 50.6 | 41.8 |
| main_b8_w24 | 61.0 | 86.9 | 46.1 | 36.4 | 72.8 | 43.4 |
| real_b2 | 54.1 | 65.8 | 35.6 | 28.3 | 32.4 | 28.1 |
| real_main_mb4 | 54.2 | 48.1 | 51.0 | 34.8 | 28.6 | 30.9 |
| real_main_mb16 | 66.9 | 58.7 | 36.9 | 40.4 | 37.4 | 33.9 |

Score→prior 反向 host wall（ms，`autograd.grad(log_prior, span_potentials)`，V3 为搜索图、V3-C 为重建图）：

| case | v3 | v3c | v3_compile_default | v3_compile_ro | v3c_compile_default | v3c_compile_ro |
|---|---:|---:|---:|---:|---:|---:|
| smoke_b2_w24 | 32.1 | 14.0 | 32.0 | 18.4 | 14.9 | 14.0 |
| smoke_b8_w24 | 24.9 | 14.4 | 33.8 | 38.3 | 15.1 | 14.3 |
| main_b1_w24 | 25.3 | 14.0 | 31.4 | 19.0 | 17.6 | 21.5 |
| main_b4_w24 | 24.8 | 14.6 | 30.1 | 38.8 | 14.4 | 15.8 |
| main_b8_w24 | 27.7 | 15.4 | 36.6 | 19.3 | 15.0 | 14.5 |
| real_b2 | 19.7 | 10.4 | 37.3 | 14.5 | 10.4 | 9.6 |
| real_main_mb4 | 16.6 | 11.7 | 20.1 | 21.0 | 10.2 | 10.1 |
| real_main_mb16 | 21.1 | 12.1 | 33.2 | 16.8 | 12.6 | 12.1 |

GPU 空闲比例（profiled 运行，kernel busy union / 步 wall）：

| case | v3 | v3c | v3_compile_default | v3_compile_ro | v3c_compile_default | v3c_compile_ro |
|---|---:|---:|---:|---:|---:|---:|
| smoke_b2_w24 | 0.94 | 0.93 | 0.94 | 0.92 | 0.93 | 0.93 |
| smoke_b8_w24 | 0.93 | 0.91 | 0.91 | 0.91 | 0.92 | 0.90 |
| main_b1_w24 | 0.88 | 0.88 | 0.88 | 0.87 | 0.88 | 0.89 |
| main_b4_w24 | 0.85 | 0.87 | 0.83 | 0.82 | 0.86 | 0.81 |
| main_b8_w24 | 0.81 | 0.79 | 0.81 | 0.79 | 0.79 | 0.78 |
| real_b2 | 0.94 | 0.93 | 0.94 | 0.93 | 0.94 | 0.93 |
| real_main_mb4 | 0.87 | 0.87 | 0.84 | 0.89 | 0.84 | 0.84 |
| real_main_mb16 | 0.85 | 0.83 | 0.82 | 0.82 | 0.82 | 0.81 |

- **V3-C（score reconstruction）单独**：中位加速 0.98–1.19，只在 batch 最小的 case 显著（`real_b2` 1.19、`main_b1` 1.10），B≥4 时 ≈1.0。它把 score→prior 反向从 25–32 ms 降到 10–15 ms（autograd 节点 892→412，与设计一致），但前向多了一段 12–20 ms 的重建 CUDA envelope 与一次额外 H2D（`cudaMemcpyAsync` 188→422 次/步：plan 索引上传）。重建本身的 kernel 很小；多出来的时间几乎全是发射与同步开销，正好抵消反向省下来的发射开销。**结论：在 CPU-bound 的现状下，V3-C 省掉的是 autograd 的 CPU 开销而不是 GPU 时间，净收益 ≈0–20%。** 它的真正价值要等搜索被编译/CUDA graph 吃掉发射开销之后才会在大 batch 上体现——但本轮测量里并没有（见下一条）。
- **compile default（inductor，无 CUDA graph）**：1.06–1.22。搜索 launch 1698→762/步（inductor 把每个 width 的 gather/pair/mask/topk 融成少数 kernel），搜索前向 54–71 ms → 35–51 ms；但 Dynamo 的 guard 检查与 Python 入口开销让每个 width 仍需约 1.5–2 ms。
- **compile reduce-overhead（每 width 一张 CUDA graph）**：**1.13–1.46，八个 case 中六个最快**。每步 34–46 次 `cudaGraphLaunch`（V3 带梯度时每个 width 2 张：AOTAutograd 的前向图与反向图各一张；V3-C 在 `no_grad` 下只有前向图，17–23 次）；搜索 launch 622/步，搜索前向降到 28–42 ms。剩余的 28–42 ms 不是 kernel（搜索 kernel 总共 ≈3–5 ms），而是 23 个 width 各自的 Dynamo guard + graph replay + eager `torch.cat(bank)`/bands 拼接 + 下一 width 的索引准备，这部分在 Python 层，CUDA graph 无法吞掉。
- **V3-C + compile**：不叠加。`v3c_compile_ro` 1.12–1.37，与 `v3_compile_ro` 持平或略慢（除 `smoke_b2`、`main_b1` 两个最小 case 略快）。原因同上：搜索发射开销被 CUDA graph 吃掉后，V3-C 多出的重建 H2D/launch 开销（~12–20 ms envelope）与它省下的反向开销（~10–20 ms）相抵。
- **GPU 仍然空闲 78–94%**。没有任何 arm 改变 GPU busy（56–98 ms/步，与 V3 相同 ±2 ms）；所有加速都来自 CPU 侧减少发射与 autograd 开销。这与 §1 的审计一致：这台 GPU 上整个 step 的 kernel 只需 60–100 ms，而 step 需要 240–310 ms。
- 峰值显存六个 arm 相同（±0.01 GiB）。

> **注意**：§4.1–4.6 的编译 arm 使用第一版形状策略（静态 shape、非 tree CUDA graph，§3）。最终配置（符号行维 + cudagraph trees）在 §4.7 复测，训练对比（§6）使用最终配置。

### 4.5 编译成本（不隐藏）

| case | arm | 首次调用 ms（含编译） | 第二次调用 ms | graph breaks | 计时期间重编译 | cudagraph skips | 每步 CUDA graph launch | 回本所需 micro-batch 步数 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| main_b1_w24 | v3_compile_default | 170.7 s | 251 | 0 | 0 | 0 | 0 | 3363 |
| main_b1_w24 | v3_compile_ro | 45.3 s | 239 | 0 | 0 | 0 | 46 | 634 |
| main_b1_w24 | v3c_compile_default | 44.2 s | 282 | 0 | 0 | 0 | 0 | 951 |
| main_b1_w24 | v3c_compile_ro | 30.5 s | 259 | 0 | 0 | 0 | 23 | 452 |
| main_b4_w24 | v3_compile_default | 172.2 s | 262 | 0 | 0 | 0 | 0 | 11586 |
| main_b4_w24 | v3_compile_ro | 46.2 s | 250 | 0 | 0 | 0 | 46 | 1206 |
| main_b4_w24 | v3c_compile_default | 43.8 s | 286 | 0 | 0 | 0 | 0 | 2186 |
| main_b4_w24 | v3c_compile_ro | 31.4 s | 247 | 0 | 0 | 0 | 23 | 1129 |
| main_b8_w24 | v3_compile_default | 9.9 s | 278 | 0 | 0 | 0 | 0 | 514 |
| main_b8_w24 | v3_compile_ro | 6.1 s | 259 | 0 | 0 | 0 | 46 | 123 |
| main_b8_w24 | v3c_compile_default | 4.7 s | 332 | 0 | 0 | 0 | 0 | 190 |
| main_b8_w24 | v3c_compile_ro | 4.4 s | 254 | 0 | 0 | 0 | 23 | 117 |
| real_b2 | v3_compile_default | 126.1 s | 146 | 0 | 0 | 0 | 0 | 2508 |
| real_b2 | v3_compile_ro | 35.3 s | 170 | 0 | 0 | 0 | 38 | 536 |
| real_b2 | v3c_compile_default | 28.4 s | 149 | 0 | 0 | 0 | 0 | 659 |
| real_b2 | v3c_compile_ro | 23.0 s | 136 | 0 | 0 | 0 | 19 | 370 |
| real_main_mb16 | v3_compile_default | 152.3 s | 243 | 0 | 0 | 0 | 0 | 6857 |
| real_main_mb16 | v3_compile_ro | 40.1 s | 258 | 0 | 0 | 0 | 40 | 1411 |
| real_main_mb16 | v3c_compile_default | 42.8 s | 277 | 0 | 0 | 0 | 0 | 4629 |
| real_main_mb16 | v3c_compile_ro | 27.0 s | 248 | 0 | 0 | 0 | 20 | 1140 |
| real_main_mb4 | v3_compile_default | 111.4 s | 221 | 0 | 0 | 0 | 0 | 4158 |
| real_main_mb4 | v3_compile_ro | 31.6 s | 204 | 0 | 0 | 0 | 34 | 1031 |
| real_main_mb4 | v3c_compile_default | 24.7 s | 238 | 0 | 0 | 0 | 0 | 1265 |
| real_main_mb4 | v3c_compile_ro | 20.6 s | 203 | 0 | 0 | 0 | 17 | 634 |
| smoke_b2_w24 | v3_compile_default | 204.2 s | 176 | 0 | 0 | 0 | 0 | 6444 |
| smoke_b2_w24 | v3_compile_ro | 49.1 s | 212 | 0 | 0 | 0 | 46 | 1485 |
| smoke_b2_w24 | v3c_compile_default | 59.5 s | 178 | 0 | 0 | 0 | 0 | 1527 |
| smoke_b2_w24 | v3c_compile_ro | 33.5 s | 164 | 0 | 0 | 0 | 23 | 710 |
| smoke_b8_w24 | v3_compile_default | 201.8 s | 203 | 0 | 0 | 0 | 0 | 18270 |
| smoke_b8_w24 | v3_compile_ro | 49.9 s | 178 | 0 | 0 | 0 | 46 | 1318 |
| smoke_b8_w24 | v3c_compile_default | 59.9 s | 179 | 0 | 0 | 0 | 0 | 2325 |
| smoke_b8_w24 | v3c_compile_ro | 33.5 s | 170 | 0 | 0 | 0 | 23 | 797 |

- 首次调用成本依赖磁盘上的 inductor 缓存：本次 benchmark 使用全新的 `TORCHINDUCTOR_CACHE_DIR`，每个 case 的第一个编译 arm（`v3_compile_default`）支付冷编译 **110–205 s**（23 个 width 的前向+反向共 46 张图，这台 CPU 上每张 2–4 s），同 case 后续 arm 与后续 case 复用缓存的 Triton kernel，降到 4–60 s；`main_b8_w24` 整体只需 4–10 s 是因为它与 `main_b4_w24` 的 width 图形状完全相同（rows 不同但 FX 图缓存命中）。
- 回本步数 = (首次 − 第二次) / 每步节省。`v3_compile_ro` 在冷编译下需 120–1200 个 micro-batch 步；按 §1.3 的真实 micro-batch 数（mb4 ≈ 86、mb8 ≈ 43 个 micro-batch / update）即 **3–30 个 update**，相对任何正式训练（5M token ≈ 1220 update）可忽略；但每个新的 (rows, words) 形状都要再编一组 23 张图（见 §4.6 stream 结果）。
- graph break 0、计时期间重编译 0、cudagraph skip 0。`compiled_frames` 列在 case 级别为 0 是因为 Dynamo 计数器在每个 case 开始时被重置而首次调用在 warmup 内完成；stream 测量（§4.6）直接报告每个 arm 编了多少帧。

### 4.6 真实流（连续真实 micro-batch，形状变化，含全部重编译）

`benchmark_cky_v35.py --stream --stream-micro-batch 4 --stream-steps 30`：30 个连续真实 micro-batch（探针 seed 7350001，跳过前 3 个 update），11 种不同 (rows, words) 形状（rows 固定 16 = 4 history × 4，words 11–20）。每个 arm 从同一快照恢复，编译 arm 使用全新 `TORCHINDUCTOR_CACHE_DIR`（冷）。结果 `reports/cky_v35/benchmark/stream/stream_mb4.{csv,json}`：

| arm | 30 步总 wall | 首步 | 中位步 | 最后 1/4 中位步 | 相对 V3（最后 1/4） | Dynamo 图数 | FX 图编译数 | 编译占用（>2 s 的步之和） |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v3 | 8.4 s | 932 ms | 254.5 | 251.6 | 1.00 | 0 | 0 | 0 |
| v3c | 7.2 s | 289 ms | 242.9 | 243.5 | 1.03 | 0 | 0 | 0 |
| v3_compile_default | **1337 s** | 170 s | 260.8 | 223.8 | 1.12 | 182 | 364 | 1333 s |
| v3_compile_ro | **411 s** | 40 s | 248.8 | 216.5 | 1.16 | 182 | 364 | 407 s |
| v3c_compile_default | 343 s | 48 s | 263.4 | 222.8 | 1.13 | 182 | 182 | 339 s |
| v3c_compile_ro | 268 s | 25 s | 234.6 | 204.4 | **1.23** | 182 | 182 | 264 s |

- `dynamic=False` 下每个 (words, width) 对是一张独立的 Dynamo 图：11 种 words × (words−1) 个 width = **182 张图**；带梯度的 V3 还要为每张图编译反向 FX 图（364 个 FX 图），V3-C 在 `no_grad` 下只编前向（182 个），因此 V3-C 的编译成本约为 V3 的一半到四分之一（后者还受益于前者留下的磁盘缓存）。
- 30 步里有 11 步触发编译（每个新 words 一次，每次 15–185 s）。在这台 CPU 上 **冷编译总成本 264–1333 s，是 30 步 eager 计算（8.4 s）的 30–160 倍**。只看稀疏的稳态步（最后 1/4，形状都已见过），加速 1.12–1.23，与 §4.3 的单形状 paired benchmark 一致。
- 训练中 words 的取值只有 ≤24 个（真实数据 9–24，§1.3），rows 在固定 micro-batch 下只有 1–2 个值（末尾不满 batch）；因此一个训练进程大约需要 ~2×24×23 ≈ 1100 张图（含反向），冷编译估计 **30–80 分钟**（按本次 11 形状 364 图 1333 s 外推），之后不再重编译。磁盘缓存（`TORCHINDUCTOR_CACHE_DIR`）可以跨进程复用，使后续运行降到数分钟。**这不是本机 CPU 的特例，Blackwell 机器的编译同样在 CPU 上进行**，需要在那台机器上重新测量。
- 几何缓存：每个 arm 14 次 miss（11 种形状 + 3 次 warm-up 清空后重建），命中率 53%（30 步中 16 次命中），构建 75–95 ms/arm（第一个 arm 255 ms，含 Python 预热）；以 mb4 每 update ≈86 个 micro-batch 计，构建成本可忽略，与 §1.4 一致。
- mb8/mb16 的流测量未单独进行：mb16 在本机会溢出到共享内存（§0），mb8 的含编译端到端对比由 §6 的训练验证给出（同样包含全部编译时间）。

### 4.7 最终配置复测（符号行维 + cudagraph trees，`reports/cky_v35/benchmark_dynamic_rows/`）

Stage E 失败后（§3 第二版）编译 arm 的 kernel 与运行时都变了，因此在 5 个 case 上按同一协议重跑了 paired benchmark（V3 eager 重新作为同进程参考）。全部 20 个非参考行：搜索身份 True、RNG 相同、logits/loss 逐位相等、梯度严格失败 0、graph break 0、计时期间重编译 0、cudagraph skip 0；每步 `cudaGraphLaunch` 数与第一版完全相同（v3 46/46/38/34/40，v3c 23/23/19/17/20），trees 正常重放。汇总在 `benchmark.csv`（列 `compile_shape_policy`）与 `compile_analysis.json` 的 `static_vs_dynamic_rows_rebenchmark`。

Paired 中位加速（第一版静态图 → 最终版符号行维）：

| case | v3_compile_default | v3_compile_ro | v3c_compile_default | v3c_compile_ro |
|---|---:|---:|---:|---:|
| main_b4_w24 | 1.064 → 1.019 | 1.163 → 1.145 | 1.076 → 1.072 | 1.118 → 1.119 |
| main_b8_w24 | 1.067 → 1.029 | 1.202 → 1.129 | 1.063 → 1.076 | 1.153 → 1.147 |
| real_b2 | 1.220 → 1.079 | 1.461 → 1.190 | 1.284 → 1.118 | 1.372 → 1.277 |
| real_main_mb4 | 1.086 → 1.083 | 1.157 → 1.108 | 1.041 → 1.106 | 1.149 → 1.121 |
| real_main_mb16 | 1.075 → 1.091 | 1.128 → 1.079 | 1.106 → 1.109 | 1.159 → 1.105 |

- `v3_compile_ro` 的收益系统性下降（1.13–1.46 → **1.08–1.19**）：搜索前向 host wall 从 28–40 ms 回升到 35–56 ms。原因是符号行维的 guard 求值（每个 width 的 SymInt 守卫）与 trees 的路径检查/录制逻辑都比非 tree 静态图的 replay 重，而 CUDA graph 数相同、GPU busy 相同（±1 ms）。`v3c_compile_ro` 受影响较小（**1.10–1.28**），因为它每 width 只有一张前向图、没有反向 trees 节点。compile_default 的变化在噪声内。
- `real_b2` 的 V3 参考中位数两次分别为 203 ms 与 172 ms（不同进程、不同时段），说明跨会话的绝对毫秒数不可比，只能看同进程 paired 比值；这也是 §6 训练对比（各 arm 独立进程）必须考虑的噪声量级。
- 结论：**真实训练必须付出的符号行维 + trees 代价约为静态图收益的 1/3–1/2**；§4.3 的静态图数字是"形状固定时的上界"，不是训练里能拿到的数字。训练里的实际值见 §6。

## 5. Stage D：静态索引缓存与其它低层项目（`stage_d_cky_v35.py`，输出 `reports/cky_v35/stage_d/`）

Stage A 的证据只支持两项：缓存构建成本（§1.4：真实 update 的 4–5%）与 AdamW 的 GPU 占比（§1.2）。长度分桶（会改变 micro-batch 的组成，需要训练验证）与历史句去重（≤0.7%）**未实现**，理由见 §1.3/§1.5。

### 5.1 向量化几何构建器（`CKYIndexCache.fast_build`，默认开启；`_build_reference` 保留，`fast_build=False` 可切回）

索引平面只依赖 (words, k, r)，行数只是一个线性偏移：新构建器先算一次与行数无关的"单位几何"（按 (words,k,r,device) 缓存），再用一次广播生成任意行数的 id_bank/decode/layout。`tests/test_index_cache_fast.py` 对 7 组 (words, rows, k, r) × {full_decode 开/关} 共 14 个 CPU 用例加 1 个 CUDA 用例断言与原 Python 构建器**逐张量逐位相同**（包括 −1 哨兵列与 layout 顺序），Stage D 对探针流观察到的全部 136 个 (words, rows) key 再次断言相同。

| words | rows | 原 Python 构建 ms（min / median） | 向量化构建，首个 (words,k,r) 冷 ms | 向量化构建，几何已缓存 ms（min / median） | 设备字节 |
|---:|---:|---:|---:|---:|---:|
| 13 | 4 | 6.7 / 7.1 | 10.1 | 3.9 / 4.1 | 250,368 |
| 18 | 16 | 16.2 / 20.8 | 25.1 | 5.7 / 6.7 | 2,390,864 |
| 24 | 16 | 36.5 / 41.3 | 41.7 | 7.2 / 7.4 | 6,752,704 |
| 24 | 64 | 45.8 / 47.8 | 38.2 | 8.9 / 9.2 | 23,860,672 |
| 24 | 128 | 65.3 / 66.6 | 38.4 | 7.3 / 8.0 | 46,671,296 |
| 24 | 512 | 181.4 / 189.5 | 39.2 | 7.0 / 7.4 | 183,535,040 |

- 原构建器随行数线性增长（w=24：n=16 需 37–41 ms，n=512 需 181–190 ms）；向量化构建器与行数无关，w=24 稳定在 **6.5–7.5 ms**（偶发 11–13 ms 为 GPU 分配抖动），首个 (words,k,r) 另付 ~37 ms 一次性的单位几何构建。缓存驻留字节不变（相同张量）。
- 它不改变任何搜索输出（plan 逐位相同），只改变 shape-only 的几何生成方式。

### 5.2 容量 8 / 32 / 64 的真实流重放（40 个连续真实 micro-batch，lazy_tensorized eager，warmup 后清空缓存）

| micro-batch | 不同 key | 容量 | 构建器 | 40 步总 wall s | 中位步 ms | miss / hit | 驱逐 | 构建总 ms | 构建占比 | 缓存驻留 MB |
|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 4 | 12 | 8 | reference | 10.48 | 259.1 | 16 / 24 | 8 | 292 | 2.8% | 29.8 |
| 4 | 12 | 8 | fast | 10.35 | 251.0 | 16 / 24 | 8 | 281 | 2.7% | 29.8 |
| 4 | 12 | 32 | fast | 10.79 | 260.0 | 12 / 28 | 0 | 266 | 2.5% | 33.8 |
| 4 | 12 | 64 | fast | 10.49 | 252.9 | 12 / 28 | 0 | 265 | 2.5% | 33.8 |
| 8 | 9 | 8 | reference | 11.33 | 282.1 | 11 / 29 | 3 | 309 | 2.7% | 53.1 |
| 8 | 9 | 8 | fast | 11.24 | 277.8 | 11 / 29 | 3 | 255 | 2.3% | 53.1 |
| 8 | 9 | 32 | fast | 11.14 | 276.6 | 9 / 31 | 0 | 244 | 2.2% | 59.3 |
| 8 | 9 | 64 | fast | 10.93 | 270.8 | 9 / 31 | 0 | 252 | 2.3% | 59.3 |

- 容量 8→32 消除全部驱逐（mb 4：8 次→0；mb 8：3 次→0），miss 从 16→12 / 11→9；但构建总量只从 ~290 ms 降到 ~265 ms（40 步 10.5 s 中的 **0.3%**），总 wall 的差异（10.3–10.8 s）在本机 CPU 噪声之内，**没有可测出的端到端收益**。§1.4 看到的 4–5% 构建占比来自 100-update 探针流里更长尾的 key 分布与每个 update 末尾的余数 batch；在 40 步窗口里只出现 9–12 个 key。
- 结论：缓存不是瓶颈。向量化构建器把每次 miss 的成本从 30–190 ms 压到 ~7 ms（大行数时 ≥10×），已足够；LRU 容量保持 8 不变（可配置）。不建议再在缓存上投入。

### 5.3 AdamW 诊断（`optimizer_check.json`，只测量，不改默认）

同参数、同梯度、同动量，各跑 8 步：foreach（当前）CUDA 中位 61.6 ms，`fused=True` 10.7 ms；8 步后参数最大绝对差 1.9e-6（相对差在接近 0 的参数上放大到 3.8），**不是逐位相同**，因此按本轮规则不采用。更重要的是：`train_update` 对一个 update 的全部 micro-batch（mb 8 时 ≈44 个）累积梯度后只做**一次** clip+AdamW，所以 §1.2/§4 里"每 micro-batch 步"包含的 clip+AdamW（≈38 ms）在真实训练里是按 update 摊销的（≈0.3% 的 update 时间）。换 fused AdamW 对训练吞吐的影响 <0.3%，不值得为之放弃逐位可复现。

**对 §4/§7 解读的修正**：benchmark 的"步"= fwd+bwd+clip+AdamW，比训练中的 micro-batch 多出 ≈38 ms 的固定项。去掉这一项后，各 arm 的相对加速略高于 §4.3 的数字（例如 `main_b8_w24` 的 `v3_compile_ro`：297→250 ms 对应 1.19，扣除 38 ms 后 259→212 ms 对应 1.22），森林在训练 micro-batch 中的占比也从 51% 升到 ≈57%。

## 6. 短程训练验证（`train_cky_v35_validation.py`，输出 `reports/cky_v35/training/`）

协议：真实 TinyStories（`cache/tinystories`），seed 42 初始化（三个 arm 的初始权重 sha256 相同，`632de5d5…`），固定 4096-token update × 100、micro-batch 8（≈44 个 micro-batch / update，末尾截断），同一验证集（50 个 batch，5279 个目标 token）每 20 个 update 评估一次；三个 arm 顺序各跑一个独立进程（`v3` → `v3_compile_ro` → `v3c_compile_ro`），编译 arm 使用全新的冷 `TORCHINDUCTOR_CACHE_DIR`。`train_wall` = `train_update` 的同步墙钟（采样、整理、H2D、forward、backward、clip、AdamW、CPU 调度与 GPU 等待），不含验证与 checkpoint IO。原始：`training/<arm>/{metrics.csv,summary.json}`；汇总：`training_throughput.csv`、`training/training_curves.csv`、`figures/training.png`。

### 6.1 吞吐（真实数据，含全部编译成本）【测量】

| arm | 100 update 训练墙钟 | token/s（含编译） | 首个 update | 编译/重编译尖峰（update 编号） | 编译开销估计 | 稳态 update 中位（后 50 个） | 稳态 token/s | 稳态相对 V3 | 后 50 个 update 含重编译相对 V3 | 回本 update 数 |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| v3（eager lazy_tensorized） | 1107 s | 370.0 | 11.0 s | 无 | 0 | 10.92 s | 375 | 1.00 | 1.00 | — |
| v3_compile_ro | 2524 s | 162.3 | 1176 s | 1, 2, 4, 6, 13, 68, 81 | 1469 s | 10.39 s | 394 | **1.05** | 0.93 | ≈2800 |
| v3c_compile_ro | 1602 s | 255.7 | 499 s | 1, 2, 4, 6, 13, 68, 81 | 615 s | 9.83 s | 417 | **1.11** | 1.05 | ≈560 |

- 编译开销估计 = 超过稳态中位 2 倍的 update 中超出部分之和；尖峰 update 都对应首次出现的 words 形状（行维已符号化，行数变化不再重编，§3/§4.7）。`v3_compile_ro` 每张图还带反向图，所以编译成本约为 `v3c_compile_ro` 的 2.4 倍。`compiled_frames` 列在本次训练里记录为 0——脚本读取了 torch 2.12 里不存在的 `frames` 计数器（训练后已改为 `stats.unique_graphs`，未重跑），图数只能由尖峰推断：7 个尖峰 ≈ 7 种 words 形状 × 23 width（带梯度再 ×2）。
- **稳态加速只有 1.05× / 1.11×**，明显低于 §4.3 单形状 paired benchmark 的 1.13–1.46，也低于 §4.7 最终配置复测的 1.08–1.28。差距来源：(a) §4.7 已测得符号行维 + cudagraph trees 本身比静态图慢（搜索前向多 5–15 ms）【测量】；(b) 训练 update 含采样/整理/H2D 与 ≈44 个形状各异的 micro-batch，截断的末尾 batch 每个 update 都带来一个新行数，trees 对新行数要重新录制而不是重放【推导，录制次数未单独计数】；(c) 训练口径下 clip+AdamW 每 update 只做一次而 benchmark 每个 micro-batch 都做（§5.3），这一项会让训练口径的比值更高而不是更低，所以不是原因。
- 100 个 update 内编译 arm 的总墙钟都**慢于 V3 eager**（0.44×、0.69×）。按稳态节省（每 update 0.53 s / 1.09 s）推算，`v3_compile_ro` 需 ≈2800 个 update 才能回本，`v3c_compile_ro` ≈560 个；5M-token 正式训练 ≈1220 个 update，前者净亏，后者净收益约 (1220−560)×1.09 s ≈ 12 分钟（≈5%）【推导】。预热的磁盘 inductor 缓存可把首个 update 降到数分钟（§4.5），回本点相应前移，但 Dynamo 守卫/trees 录制部分仍在。
- 峰值显存 3.37 / 3.36 / 3.36 GiB，trees 的共享内存池没有增加峰值；几何缓存 miss 均为 1095（同一数据流）。

### 6.2 训练曲线与行为指标【测量】

| update | val NLL v3 | v3_compile_ro | v3c_compile_ro |
|---:|---:|---:|---:|
| 0 | 9.819713 | 9.819713 | 9.819713 |
| 20 | 7.429117 | 7.428993 | 7.428943 |
| 40 | 6.319617 | 6.319515 | 6.319689 |
| 60 | 5.199266 | 5.199745 | 5.199434 |
| 80 | 4.435981 | 4.436220 | 4.435403 |
| 100 | 4.021152 | 4.020780 | 4.020440 |

| arm | 最后 10 个 update 训练 NLL | 平均梯度范数 | span scorer 梯度范数 | 末尾候选熵 | 根候选数 | 非有限梯度 |
|---|---:|---:|---:|---:|---:|---:|
| v3 | 4.4444 | 2.1781 | 0.0079 | 0.913 | 3.98 | 0 |
| v3_compile_ro | 4.4452 | 2.1801 | 0.0088 | 0.907 | 3.97 | 0 |
| v3c_compile_ro | 4.4439 | 2.1791 | 0.0100 | 1.076 | 3.97 | 0 |

- 第 1 个 update 三个 arm 的训练 NLL（9.82033599447459）与梯度范数（8.624692916870117）完全相同：同一初始权重上的 forward 与第一次反向在打印精度内一致，与 §4.1 的逐位身份结论一致。从第 2 个 update 起轨迹在第 5–6 位有效数字开始分叉，100 个 update 后 val NLL 曲线最大差 4.8e-4 / 7.1e-4，终值差 −3.7e-4 / −7.1e-4（都略低于 V3，在噪声内，见 6.3）。

### 6.3 分叉的归因：V3 eager 自身不可逐位复现【测量】

为判断分叉是否由编译引起，另跑了三个 3-update 短程（`training/determinism_check/`）：V3 eager 两次（同 seed、同数据顺序）在第 2 个 update 的训练 NLL 分别为 9.745971988886597 与 9.746007300913334，Stage E 的 V3 为 9.746016681194304；V3-C eager 为 9.745997212827206；两个编译 arm 为 9.74593724310398 / 9.745964787900448。即 **V3 eager 自身两次运行之间的差异（3.5e-5）与编译 arm 和 V3 之间的差异（8e-5）同量级**。原因是 eager CUDA 反向里 `index_select`/`scatter_add`/embedding 的原子加法顺序不确定（`torch.use_deterministic_algorithms` 未开启；开启会改变训练默认实现，本轮不做）。因此训练轨迹层面无法验证、也不应要求任何 arm 与 V3 逐位相同；正确的身份检验是 §2.2/§3/§4.1 的"同一权重、同一 micro-batch"逐位比较（已全部通过），训练层面能确认的只是三条曲线的差异落在 V3 自身不可复现范围内。

### 6.4 回归测试

最终代码上 `pytest -q tests`：**513 passed, 1 skipped**（`tests/test_forest_lazy.py` 单词句无二词子 span 的既有跳过）；其中 `tests/test_cky_compile.py` 11 个（含 `test_dynamic_rows_do_not_recompile_and_match_eager`）、`tests/test_score_reconstruction.py`、`tests/test_index_cache_fast.py` 与全部旧实现（score_first / lazy_semantic / lazy_tensorized）测试。

## 7. 性能上限估计

全部基于 §1.2（真实 mb 8 micro-batch，V3 eager）与 §4（paired benchmark）的本机测量；标注【测量】【推导】【理论界】【未验证】。目标 GPU 上必须重测后再做同样的推导。

### 7.1 一个 micro-batch 的时间去向（真实 mb 8，V3 eager，ms）【测量】

| 块 | CUDA envelope | kernel active | 性质 |
|---|---:|---:|---|
| score-only CKY search（23 width） | 58.9 | 3.0 | CPU 发射 bound |
| Lazy tail（D2H + CPU DAG + H2D） | 5.7 | ≈0 | CPU |
| Composer forward（width-batched） | 17.3 | 0.9 | CPU 发射 bound |
| CKY backward（staged，含 Composer bwd 33.7） | 84.1 | 2.2 (+1.9 Composer) | CPU autograd/发射 bound |
| **森林合计** | **166.0** | **≈8** | 51% 的 step |
| syntax fwd + bwd | 22.0 | ≈2 | CPU bound |
| decoder+readers fwd + bwd | 38.1 + 54.3 | 4.1 + 7.4 | CPU bound |
| clip | 7.6 | 4.3 | 混合 |
| AdamW（foreach） | 30.9 | 30.9 | **GPU bound** |
| 未归属 / 空隙 | ≈8 | — | — |
| **step（未加仪器）** | **326.6** | **62.2** | GPU 空闲 81–86% |

### 7.2 Amdahl 推导【推导】

- 森林部分占 benchmark step 的 f ≈ 0.51（166/327）；扣除训练里按 update 摊销的 clip+AdamW（§5.3）后，占训练 micro-batch 的 f ≈ 0.57（166/289），下面的推导用 benchmark step 口径，训练口径的天花板相应略高（≈2.2×）。把森林部分压到它的 kernel 时间（≈8 ms）是本轮方法（不改算法、不写 kernel）的理论下界：step ≈ 327 − 166 + 8 ≈ 169 ms，**完整 update 的最大可能加速 ≈ 1.9×**。这是 mb 8 在这台 GPU 上"森林优化"的天花板；再往上必须动 decoder/reader 的发射开销（与 CKY 无关）或改 AdamW（已是 GPU bound，改 fused 只省 ~0 ms kernel）。
- 本轮实际达到：`v3_compile_ro` 在 mb 4–16 级别的 case 上 1.13–1.20（§4.3），即从 166 ms 森林时间里去掉了约 40–50 ms（搜索 59→36、搜索反向 25→19）。**达到了森林天花板（1.9×）的约 1/4**。训练口径（§6）更低：最终配置在真实训练稳态只有 1.05×（`v3c_compile_ro` 1.11×），即森林时间只去掉了约 15–30 ms，不到天花板的 1/6。
- 剩余的森林开销（compile_ro 之后，mb 8）：搜索 36 ms（其中 kernel 3–5 ms；剩余是 23 个 width 各自的 Dynamo guard、CUDA graph replay、eager `cat`/bands 与索引准备）；CKY backward 中 Composer backward ≈34 ms 与 Composer forward ≈17 ms（本轮未触碰，kernel 合计 <3 ms）；score backward 15–19 ms；tail 6 ms。这四项 ≈ 110 ms 中 ≥95% 仍是调度开销。
- 把森林天花板外推到整个 step：step 中 GPU 真正忙 62 ms，其中 AdamW 31 ms。如果 forward/backward 的全部 CPU 开销都被隐藏（整步 CUDA graph 或持久 kernel——本轮明确不可行，因为 Lazy DAG 需要 D2H+CPU 闭包），step 下界 ≈ 62–70 ms，**整步理论上限 ≈ 4.7–5×**【理论界】。

### 7.3 哪些是真正不可避免的语义森林计算【测量 + 推导】

- 搜索 + Composer fwd/bwd + CKY backward 的 kernel active 合计 ≈ 8–12 ms / micro-batch（mb 4–16），占 step 的 2.5–3.5%。这是 K=4、R=12、W≤24、H=4 下这台 GPU 的真实森林算术量；其余 ≈155 ms 全是可消除的调度/发射/autograd 开销。
- 随 micro-batch 增大，launch 数几乎不变而 kernel 时间按比例增长（§1.2：mb 4→16 时 busy 61→79 ms，wall 335→348 ms）。按此斜率外推，森林 kernel 要和 CPU 发射时间（~160 ms）打平需要 mb ≈ 300–500——本机显存不允许，**在 RTX PRO 6000 上也只有在 micro-batch ≥64–128 时森林计算才可能成为主要成本**【未验证】。更快的 GPU 只会缩短 kernel 时间，CPU 发射时间不变，因此在目标机器上 GPU 空闲比例预计更高而不是更低。

### 7.4 可能达到的加速 vs. 不可能达到的加速【推导】

| 目标 | 估计 | 依据 |
|---|---|---|
| compile_ro（已实现，已验证） | 静态形状 benchmark 1.13–1.46；最终（符号行维）配置 1.08–1.19；真实训练稳态 1.05（V3-C + ro 1.11） | §4.3 / §4.7 / §6 测量 |
| 再编译 Composer 前向/反向（纯张量 width-batched，与搜索同类） | 再省 ≈30–40 ms/步（≈ +15%） | Composer 51 ms envelope 中 kernel 2.8 ms；**未实现未验证** |
| V3-C + compile_ro 合并上传 plan（一个 pinned H2D 代替 ~230 次小拷贝） | 使 V3-C 的反向节省（10–15 ms）真正叠加 | §4.4：V3-C 多出 234 次 memcpy；**未实现未验证** |
| 长度分桶（减少 width 数） | ≈ −11 width ≈ −70–80 ms/步（22%）**仅当** launch 数随 width 线性下降 | §1.3 估计；改变 micro-batch 的组成，需要训练验证；本轮未实现 |
| fused AdamW | ≈0 ms（AdamW 已经 GPU bound） | §5 诊断 |
| 整步 CUDA graph | 不可行（Lazy DAG 的 D2H/CPU 闭包） | §3 |
| Triton/CUDA 融合 kernel（下一轮） | 只能缩短 3–12 ms 的 kernel 时间，**对现状几乎无用**；它的价值在于把每个 width 的 ~50 个 launch 合成 1–3 个 | §1.2 launch 计数 |

**结论**：在 K=4/R=12/W≤24/H=4 与 mb ≤16 的设定下，这台机器上 V3 的训练时间约 97% 是可消除的调度开销，只有 ≈3% 是语义森林的算术；本轮无 kernel 的方法在固定形状 benchmark 上把 update 时间压到了 V3 的 0.69–0.89，但在真实训练稳态只压到 0.90–0.95（§6），距离"森林开销全部消除"的 1.9× 天花板还有约 1.4× 的空间，主要在 Composer fwd/bwd 的发射开销与 width 循环的 Python 层。

## 8. 最终回答

标注：【测量】本机实测；【推导】由测量数据做的性能模型推导；【理论界】估计的理论上限；【假说】尚未检验的工程假说。所有数字只在本机（RTX 3070 Laptop 8 GB，§0）内部比较。

1. **当前 V3 的主要性能瓶颈是什么？**【测量】CPU 侧的算子发射与 autograd 记录，不是 GPU 计算、不是 CPU/GPU 同步、不是数据管线。一个真实 micro-batch 约 330 ms 中 GPU kernel 只忙 60–80 ms（§1.2），5000–5700 次 launch × ~60 µs 全程开销就是步长；最大的两块是 CKY backward（84–91 ms envelope，kernel 2–4 ms）与 score search（59–69 ms envelope，kernel 3–4 ms）。
2. **CPU 提交与 GPU 真正计算分别占多大比例？**【测量】GPU kernel active 占 profiled step envelope 的 12–16%（mb 4/8/16 空闲比例 0.88/0.86/0.84）；以 busy 60.8/62.2/79.5 ms 对未加仪器 wall 334.9/326.6/347.5 ms 计为 18–23%。其余 77–88% 的时间 GPU 空闲、CPU 在发射。GPU 忙时里 AdamW 占 40–50%（训练中按 update 摊销，§5.3）；森林（搜索+Composer+CKY backward）的 kernel 仅 8–12 ms = step 的 2.5–3.5%。
3. **真实训练中静态索引缓存的命中率是多少？**【测量】LRU 8 在真实 100-update 探针流上：mb 1/2/4/8 分别 34%/51%/65%/78%，mb 16–32 约 88%，mb 64–128 回落到 73–81%（余数 batch 的新 key）；计时 update 中实测 32%/49%/66%/80%，构建占 update 墙钟 4–5%（§1.4）。容量 32 可把 mb ≤8 提到 97–99%，但 Stage D 的 40 步重放显示端到端收益在噪声内（§5.2）；向量化构建器把每次 miss 从 30–190 ms 压到 ~7 ms（§5.1，逐位相同）。
4. **Score Reconstruction 是否在保持梯度的前提下真正加速？**【测量】保持：候选 ID/backpointer/mask/RNG/根分数/log-prior/logits/loss 全部逐位相同，1080+36 身份用例与全部梯度严格容差通过（§2.2、§4.1）。加速：单独使用只在最小 batch 有 1.10–1.19，B≥4 时 ≈1.00（§4.3）；它把 score→prior 反向从 25–32 ms 降到 10–15 ms、autograd 节点 892→412，但前向多出 12–20 ms 重建 envelope 与 ~230 次小 H2D 拷贝，两者相抵。**结论：数学上成功、作为独立加速手段在当前 CPU-bound 状态下基本无效。**
5. **Score Autograd Graph 能否进一步缩小？**【测量+推导】V3-C 已把搜索阶段的梯度图缩到按 width 分层的 DAG 重建（每层 2 次 index_select + 1 次 add + cat），节点数只随 width 层数（≤23）而非候选数变化；再缩小的空间在 (a) 合并每层的 plan 上传为一次 pinned H2D【假说，预计省 5–10 ms】，(b) 把重建与 Composer plan 的 H2D 合并【假说】。继续减少节点数本身收益不大：反向剩余 10–15 ms 中 kernel <1 ms。
6. **`torch.compile` 能够成功编译哪些子图？**【测量】V3 width 循环体（gather/pair/mask/isfinite/Top-R/探索 rand+scatter/Top-K/Delta/swap/分数与 backpointer gather）作为一个 `fullgraph=True, dynamic=False` 的函数整体编译成功，graph break 0、重编译只随 (words, width, explore/swap 常量) 发生（§3、§4.5）。不能编译的部分：Lazy tail（D2H→CPU 闭包→H2D，本质上是 host 代码）、width 之间的 `torch.cat(bank)`/bands 拼接与索引准备（留在 eager）、Composer/Reader/Decoder（本轮未尝试，是下一步候选）。必要条件：`fallback_random=True`、`emulate_precision_casts=True`（负对照证明缺它会改变 width-2 的 BF16 分数，§3）、`recompile_limit=512`。
7. **`reduce-overhead` 是否实际使用 CUDA Graph？**【测量】是：每步 34–46 次 `cudaGraphLaunch`（V3 带梯度时每 width 前向+反向两张图，V3-C 只有前向 17–23 张），cudagraph skip 0（§4.5 第一版配置；最终配置复测 §4.7 的每步 graph launch 数完全相同；§6 训练里 trees 对每个新行数录制后重放）。需要两处工程处理：行维符号化后只能用 cudagraph trees（非 tree 路径要求全静态形状），并且每次搜索 forward 前调用 `cudagraph_mark_step_begin()`，否则带梯度/`no_grad` 交替时 trees 会判前一代输出失效并抛错（§3）。整步 CUDA graph 不可行（Lazy DAG 的 D2H/CPU 闭包）。
8. **编译成本需要多少训练 Step 才能摊销？**【测量+推导】单形状：冷编译 110–205 s（default）/ 25–50 s（ro，复用磁盘缓存后），每步节省 20–70 ms → 120–3400 个 micro-batch 步 ≈ **3–40 个 update**（§4.5）。真实流：每个新 words 形状都重编 23 张图（带梯度 ×2），30 步 11 形状的流冷编译 264–1333 s，是这 30 步 eager 计算的 30–160 倍（§4.6）；训练中约 24 种 words × 1–2 种 rows ≈ 1100 张图，**冷启动估计 30–80 分钟**【推导】，之后不再重编；磁盘缓存可跨进程复用。§6 实测（100 个 update，冷缓存）：`v3_compile_ro` 编译+重编译共 ≈1470 s（首个 update 1176 s，之后 6 个新 words 形状各 45–78 s），稳态每 update 只省 0.53 s → **回本 ≈2800 个 update**；`v3c_compile_ro` ≈615 s、每 update 省 1.09 s → **回本 ≈560 个 update**；100 个 update 内两者总墙钟都慢于 eager（0.44× / 0.69×）。
9. **哪个 Micro-batch 在目标 GPU 上达到最高有效训练吞吐？**【测量，仅本机】本机可靠范围内 mb 8（188 token/s）> mb 4（183）> mb 2（115）> mb 1（61）；mb 16/32 因共享内存溢出不可靠，mb 64 在两次恰好有显存的 update 上达 ≈400 token/s（下界参考），mb 128 OOM（§1.1）。**目标 GPU（RTX PRO 6000）上的最优 micro-batch 本轮无法回答**，需在那台机器上重跑 `profile_cky_v35.py --phases sweep`；按 §1.2 的斜率（launch 数与 batch 无关、kernel 时间随 batch 线性）推断最优点会在 mb ≥64【推导】。
10. **不同历史长度是否存在明显 Padding 浪费？**【测量】有：mb ≥8 时 chart 中 71–80% 的 span 是 padding，word padding 55–62%，target padding 34–49%（§1.3）。但它浪费的是 GPU 工作量（本来只占 step 的 1–3%），不是时间；只有通过减少 width 数（长度分桶）减少 launch 才能换成时间，估计 −70–80 ms/步【假说，未实现未验证，§1.3/§7.4】。历史句去重 ≤0.7%，不值得。
11. **Lazy DAG 的 CPU/GPU 同步是否已经成为主要瓶颈？**【测量】没有。D2H 前 CPU 等待 GPU 0.02–0.06 ms（GPU 早已算完），拷贝+tolist 1–3.7 ms，CPU DAG 闭包 1.9–5.5 ms，plan H2D 0.4–0.5 ms，合计 <3% 的 step（§1.2）。它的真正代价是结构性的：使整步无法进入一张 CUDA graph（Q7）。
12. **各项优化对完整 Update 的独立贡献是多少？**【测量，paired 中位数，§4.3；训练口径见 §5.3 修正】V3-C 单独 0.98–1.19；compile default 单独 1.06–1.22；compile reduce-overhead 单独 **1.13–1.46**（小模型/小 batch 高端，主模型 mb 4–16 为 1.13–1.20）；V3-C+compile ro 1.12–1.37（不叠加，与 compile ro 持平）；向量化缓存构建器 ≈0（构建占比 <1%，§5.2）；fused AdamW ≈0（按 update 摊销，且非逐位）。最终（符号行维）配置复测：compile ro 1.08–1.19、V3-C+compile ro 1.10–1.28（§4.7）。**真实训练稳态（§6，100 个 update 后 50 个中位）：compile ro 1.05、V3-C + compile ro 1.11；含编译的总墙钟 0.44 / 0.69。**
13. **最优组合是否能够保持原 CKY 搜索与梯度？**【测量】是。`v3_compile_ro`（与 `v3c_compile_ro`）在全部 8 个 benchmark case、13 个 compile 身份用例与 §6 的训练对比里：候选身份 True、RNG 相同、logits/loss 逐位相等、梯度严格容差 0 失败（§4.1、§4.7、§3）。训练轨迹层面：三条 100-update 的 val NLL 曲线最大差 7e-4，而 V3 eager 自身两次同种子运行从第 2 个 update 起就有同量级差异（eager CUDA 反向的原子加法不可复现，§6.3），所以训练层面只能确认差异在 V3 自身不可复现范围内，逐位相同既不可测也不是正确的要求。
14. **完整训练 Token/s 提高了多少？**【测量，§6，本机 mb 8、100 个 4096-token update、冷编译缓存】V3 eager 370 token/s（稳态 375）。`v3_compile_ro`：含编译 162 token/s（**−56%**），稳态 394（**+5%**）；`v3c_compile_ro`：含编译 256 token/s（−31%），稳态 417（**+11%**）。即本轮最优候选在真实训练里的稳态提升是 **5–11%**，远低于 benchmark 的 13–46%；对 ≈1220 个 update 的 5M-token 训练，`v3c_compile_ro` 净收益约 5%，`v3_compile_ro` 净亏。
15. **目前还有哪些可以量化的可消除开销？**【测量+推导，mb 8 口径】compile_ro 之后森林仍有 ≈110 ms/步的调度开销（kernel <6 ms）：Composer forward 17 ms + Composer backward 34 ms（未编译，候选收益 30–40 ms）、搜索 width 循环剩余 36 ms（Dynamo guard/graph replay/eager cat，候选：把 cat/bands 也纳入图或合并相邻 width）、score backward 15–19 ms（V3-C 的 H2D 合并后可叠加 10–15 ms）、tail 6 ms。非森林：decoder+reader fwd/bwd ≈92 ms envelope 中 kernel 11.5 ms、syntax 22 ms 中 kernel ≈2 ms——同类发射开销，可用同样的 compile 手段处理，但与 CKY 无关。
16. **是否值得进入 Triton / Custom CUDA Kernel 阶段？**【推导】以缩短 kernel 时间为目的：**不值得**——森林 kernel 合计 8–12 ms/步（3%），把它写到 0 也只提速 3%。以减少 launch 为目的（每 width ~50 次 → 1–3 次融合 kernel）：潜在收益与 compile_ro 同量级（搜索 36 ms → ~5 ms），但 compile_ro 已经拿到其中大半且零维护成本；只有在目标 GPU 上验证 (a) mb ≥64 时森林 kernel 时间确实上升到与发射时间同阶、或 (b) 编译的 30–80 分钟冷启动不可接受，才有理由进入 Triton 阶段。**建议暂不进入。**
17. **目前是否应该停止 CKY 工程优化、开始正式模型消融？**【推导】应该把 CKY 工程优化降为"背景任务"并开始模型消融：(a) 数学/搜索/梯度身份已被三种实现与两种编译后端交叉验证，继续优化不会改变模型行为；(b) 但本轮最好的候选 `lazy_tensorized_score_reconstruct + compile_reduce_overhead` 在真实训练里稳态只快 1.11×、回本需 ≈560 个 update，对 5M-token（≈1220 update）训练净收益 ≈5%；`lazy_tensorized + compile_reduce_overhead` 回本 ≈2800 update，对该规模净亏（§6）。因此**正式消融建议直接用 V3 eager（`lazy_tensorized`）**，编译配置保留为可选（`configs/v35/compile_reduce_overhead.yaml` / `score_reconstruct.yaml`），只在单次训练 ≥2000 个 update 或磁盘 inductor 缓存已预热时才值得启用；正式默认实现按规则保持 `score_first`，checkpoint 记录实现模式与 Precision Policy；(c) 余下 1.4× 的森林天花板（§7）需要编译 Composer 与合并 width 间 eager 操作，属于渐进工程，不阻塞消融；(d) 目标 GPU 上的 micro-batch 选择（Q9）是消融前唯一必须先做的测量。

### 失败与回滚记录

- 非 tree CUDA graph（第一版 `reduce-overhead` 配置）与符号行维不兼容：inductor 把标记为动态的行维重新特化为常数并抛出 `ConstraintViolationError`；改回 trees + `cudagraph_mark_step_begin()` 后通过（§3）。
- `reduce-overhead` 默认 cudagraph trees 在"V3 带梯度 → V3-C `no_grad`"的同进程交替下抛出 "accessing tensor output of CUDAGraphs that has been overwritten by a subsequent run"，整个首次 benchmark case 失败（日志保留在 `reports/cky_v35/logs/`）；改用非 tree CUDA graph 后重新通过全部身份验证（§3）。
- 不开 `emulate_precision_casts` 的编译把 width-2 BF16 叶子加法提升到 FP32，产生 2.7e-4–4.9e-4 的分数差异（负对照，`compile_identity.json`）；保留为必需设置。
- Stage A 的 micro-batch 16/32 sweep 两次落入 Windows 共享内存溢出，数据作废后用分配器封顶重跑并标注不可靠；mb 64/128 OOM（§1.1）。
- 几何快速构建器第一版把 slot 偏移错误地按 batch 缩放，eager V3 测试报 "Reachable DAG contains an invalid candidate"；修正后全部逐位测试通过。
- **Stage E 第一次尝试（静态 shape 编译）失败**：`train_update` 为了精确命中 4096 个 token 会截断每个 update 的最后一个 micro-batch，于是行数（B×H）每个 update 都不同；`dynamic=False` 下每个 (rows, words, width) 都是一张新图，`v3_compile_ro` 与 `v3c_compile_ro` 在第 6 个 update 触发 Dynamo `recompile_limit`（512）异常退出（日志 `reports/cky_v35/training/logs/train_*_static_attempt.log`；前 5 个 update 分别耗时 1384/237/152/81/151 s，几乎全是编译）。修正：把 bank/mask_bank/span_flat/left_ids/right_ids 的行维标记为符号维（`mark_dynamic`，其余维度保持静态，`automatic_dynamic_shapes=False`），width step 的运算逐行独立，RNG 消耗顺序不变。中间失败一次：只标记 bank/mask_bank/span_flat 时，`left_ids.gather` 的形状守卫把行数重新特化为常数（`ConstraintViolationError … specialized to constant (4)`），加入 left_ids/right_ids 后通过。修正后重新通过全部 compile 身份验证、复测 benchmark（§4.7）并重跑 Stage E（§6）——重跑成功但稳态只快 5–11%，编译成本在 100 个 update 内未回本。这是**静态形状编译在真实训练里的结构性问题**，不是本机特有。
- 本机的已训练主模型 checkpoint 为截断文件，无法加载；全部实验使用 seed 42 初始化（§0）。
- 流测量 mb 8 与 Stage E 以外的更大 batch 的训练对比未做（显存）。
- Stage E 的 `compiled_frames` 列记录为 0：`train_cky_v35_validation.py` 读取了 torch 2.12 中不存在的 Dynamo `frames` 计数器（流 benchmark 中同一问题已改为 `stats.unique_graphs`，训练脚本训练后才改）；图数由 update 时间尖峰推断（§6.1），未为此重跑 70 分钟的训练。
- 训练轨迹不能逐位比较：V3 eager 自身两次同种子运行从第 2 个 update 起分叉（§6.3，`training/determinism_check/`）。未开启 `torch.use_deterministic_algorithms`（会改变训练默认实现的 kernel 选择），训练层面的一致性以 val NLL 曲线差 ≤7e-4（在 eager 自身噪声内）作结。
- 真实训练收益显著低于 benchmark（§6.1）：benchmark 的 1.13–1.46 建立在静态形状、非 tree CUDA graph 上，而真实训练必须用符号行维 + trees（§4.7：收益降到 1.08–1.19）并承受每个 update 新行数的 trees 录制，稳态只剩 1.05–1.11。这是本轮最重要的"benchmark 不等于训练"的实例，已在 Q12/Q14/Q17 中如实反映。
