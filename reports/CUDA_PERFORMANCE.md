# CUDA 小批次性能剖析（2026-10-31）

此文件记录 V0 的历史测量。当前默认已启用 V2，新的同权重 Reference/V2 对比见 [CKY V2 报告](cky_v2/summary.md)。

## 结论

**当前 CKY 实现是主要瓶颈，且包含很重的 CPU 提交、张量重组和 autograd 调度开销。** 主配置在 24 词历史上的 CKY 前向/反向合计约占分阶段诊断时间的 87%；继续加快 Transformer 或 Composer 的矩阵乘法不足以解决这个瓶颈。

环境：`E:\anaconda3\envs\py312\python.exe`，Python 3.12.13，PyTorch 2.12.1+cu132，CUDA 13.2，RTX 3070 Laptop 8GB，BF16。Windows/WDDM 笔记本环境，桌面进程和功耗/温度会造成波动；每组的原始最小/最大时间及环境见 `cuda_profile/*/summary.csv`、`environment.json`。没有同时运行其它本项目 GPU 测量任务。这些结果不能直接换算成 Blackwell 吞吐量。

## 不带 profiler 的完整微批更新

每个计时包括 forward、loss、backward、梯度裁剪和 AdamW；预热 2 次、同步计时 5 次，取中位数。驻留 GPU 输入、每个 micro-batch 单独更新，无梯度累积、验证、数据加载/搬运和 Checkpoint IO。下面的 Token/s 是这种微批基准的吞吐量，不是完整训练脚本的端到端吞吐量。

除最后一行真实数据外，均为控制形状的合成输入：4 个历史句子，历史各词位置有效，目标长 128（含 EOS）。没有报告这些输入的语义准确率或 LM 研究效果。

| 配置/模式 | Batch | 每句历史词数 | 更新中位数 ms | CE Token/s | PyTorch 峰值分配 GiB |
|---|---:|---:|---:|---:|---:|
| smoke / no_memory | 2 | 不读取 | 22.99 | 11,135.7 | 0.216 |
| smoke / forest_vector4 | 2 | 8 | 187.88 | 1,362.6 | 0.413 |
| smoke / forest_vector4 | 2 | 16 | 534.44 | 479.0 | 0.515 |
| smoke / forest_vector4 | 2 | 24 | 1,212.57 | 211.1 | 0.899 |
| smoke / forest_vector4 | 8 | 24 | 1,285.63 | 796.5 | 2.707 |
| main_125m / no_memory | 1 | 不读取 | 97.21 | 1,316.7 | 2.367 |
| main_125m / forest_vector4 | 1 | 24 | 1,340.43 | 95.5 | 2.720 |
| main_125m / forest_vector4 | 4 | 24 | 1,318.43 | 388.3 | 3.453 |
| smoke / forest_vector4，真实 TinyStories 缓存 | 2 | 动态 Padding 后最长 20 | 879.59 | 33.0 | 0.643 |

真实数据行选择两个不同故事里已有四句历史的目标；两个目标共 29 个 CE Token，Padding 后目标长度 17，历史形状 `[2,4,20]`。真实文本只有很少目标 Token 可摊薄历史处理的固定成本，因此 Token/s 比控制目标长度 128 的合成输入低。这不是分词器或训练收敛的比较。

同为 24 词历史，小配置的 batch 2→8 时一步仅约慢 6%，Token/s 提高约 3.77 倍；主配置 batch 1→4 时耗时在测量波动范围内相近，Token/s 提高约 4.07 倍。显存允许时，增大 micro-batch 可显著摊薄提交成本；不能假定无限放大 batch 仍会线性提速。

## CKY 的时间分解

使用另一组运行，在 Syntax / Forest 的张量边界临时 detach，分阶段将同一条损失的梯度传回，以隔离各部分 backward。仅诊断脚本使用这种方式；训练脚本的常规单次 backward 未改变。分阶段同步和 hooks 会影响时间，以下是定位用途，不能作为无 profiler 更新的精确可加占比。

主配置，batch 1，历史 `[1,4,24]`：

| 部分 | 分阶段中位数 ms |
|---|---:|
| Syntax forward | 9.25 |
| CKY forward（包含 Composer） | 383.62 |
| 其中 Composer forward（已包含在上行） | 38.04（GPU Event 区间）/41.37（CPU 提交区间） |
| Decoder + Reader forward | 41.44 |
| Decoder + Reader backward | 57.35 |
| CKY backward | 741.33 |
| Syntax + Semantic Embedding backward | 13.98 |
| Clip + AdamW | 42.65 |

在互不重叠的诊断阶段上，完整 forward（435.49）+各 backward+优化合计约 1290.79 ms；CKY forward+backward 约 1124.95 ms，占约 87.15%。小配置 batch 2/24 词对应约 94.67%。Composer 是 CKY 的子项，Reader 是 Decoder 的子项，不能重复相加。

## 算子与代码证据

主配置一次被 profiler 跟踪的 CKY forward 范围出现：

- `torch.stack` / `aten::stack` **2258 次**。代码每个 width、每个 split 都重新从 Python Cell 列表 stack 左右的 score、mask、semantic、coeff，再 stack 局部势和根。
- `aten::cat` **2436 次**（包括 stack 的内部 cat），`aten::select` **2256 次**。
- `cudaLaunchKernel` **5372 次**。Profiler 中这些 launch API 的 CPU self time 约 171 ms；profiler 本身有开销，不能直接把这个数当作正常运行时间。
- 整个训练 trace 有 **1989 个 `SelectBackward0`**、**14168 次 `aten::add_`**。结合 CKY 独立 backward 的约 741 ms，表明 chart 索引与梯度累积产生大量小节点/小 Kernel。

另一次 Python cProfile 的主配置 forward 总计约 560 ms，Forest 占约 481 ms；`torch.stack` 累积约 157 ms，Composer 约 39 ms。这些是 cProfile 下的主机侧耗时，支持瓶颈来自结构组织和算子提交的判断，不能当作纯 GPU 算力时间或正常吞吐量。

代码中 `expand()` 虽然产生 view，后续合并复制维度的 `flatten()`、拼接、gather 仍会物化大量数据。当前“最多组合 12 个候选”只限制真正送入 Composer MLP 的数量；在此之前仍为所有 split×K² 构造/拼接语义向量与系数。这个预算没有消除这些张量搬运和反向传播工作。

## 优化优先级

1. 将 chart 从逐 Span 的 Python Cell 组织改为按 width 的整块张量，批量索引相邻区间，减少几千次 stack/select 以及对应 backward 节点。需要保留现有精确枚举、梯度、Mask、因果性和恢复测试。
2. 先在标量分数上做候选筛选，再 gather **选中的**左右语义/系数，避免先物化全部 split×K²×256 的向量。重点同时测 backward，不能只优化 forward。
3. 使用长度分桶和充分利用显存的 micro-batch，减少最长句导致其它短句也执行完整 Padding chart 的浪费。上表已经实测 micro-batch 的收益。
4. 再评估 tensorized CKY 的 compile/fusion。现有 Python 字典/循环路径不宜直接假定 `torch.compile` 能自动解决；尚未测量或声称这类优化的加速比。

本次没有改变 CKY 搜索、梯度或模型数学定义。新增的 `profile_performance.py` 可以重复这些测量。

## 文件与复现

精简结果保存在 `reports/cuda_profile/{smoke,main,batch8,main_batch4,real}/`。主配置与小配置额外保存算子 CSV、按阶段算子 CSV 和 Python cProfile 文本；原始 trace 分别为 `runs/cuda_profile_main/forest_vector4_b1_w24/trace.json` 和 `runs/cuda_profile_smoke/forest_vector4_b2_w24/trace.json`，各约百 MB，没有放入源码 ZIP。

```powershell
& 'E:\anaconda3\envs\py312\python.exe' profile_performance.py --config configs/smoke.yaml --batch-size 2 --lengths 8 16 24 --trace --output-dir runs/cuda_profile_smoke
& 'E:\anaconda3\envs\py312\python.exe' profile_performance.py --config configs/main_125m.yaml --batch-size 1 --lengths 24 --trace --output-dir runs/cuda_profile_main
& 'E:\anaconda3\envs\py312\python.exe' profile_performance.py --config configs/main_125m.yaml --modes forest_vector4 --batch-size 4 --lengths 24 --output-dir runs/cuda_profile_main_batch4
```

CUDA Event 区间包含 GPU 等待 CPU 提交的空档，不是纯计算时间。Profiler 嵌套范围与设备注解的统计可能重叠，不能直接求和其百分比；本报告仅用无 profiler 的计时描述更新速度，用单独的诊断与原生算子事件定位原因。
