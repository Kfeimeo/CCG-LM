# Local CNN 语义记忆对照

目的：检验小窗口、有序组合得到的局部特征，经过整句汇总后，是否已足以提供森林记忆的收益。该实验不能单独证明“语义只有邻接依赖”。本次交付基线和可执行入口，不进行正式效果训练。

## 模型

新增 `mode: seq_cnn4`。输入为已完成的历史句子，使用与原模型相同的独立 Semantic Embedding；不使用 SyntaxEncoder、CKY、Composer 或句法系数。每个历史句子单独卷积，不跨句传播。

设词向量为 $h_i^{(0)}=E_{sem}(x_i)$。每层使用 stride=1、dilation=1 的奇数核卷积：

$$
\widetilde h_i^{(l)}=b_l+\sum_{r=-\lfloor k/2\rfloor}^{\lfloor k/2\rfloor}W_{l,r}h_{i+r}^{(l-1)},
\qquad h_i^{(l)}=\operatorname{LN}\left(h_i^{(l-1)}+\operatorname{GELU}(\widetilde h_i^{(l)})\right).
$$

卷积前、每层后以及输出投影后均清零 padding。LayerNorm 仅沿单个词的通道维归一化，避免从 padding 或其他词引入隐藏依赖。可选输入/输出投影调整隐藏通道宽度。

汇总前的有效感受野为 $r=1+L(k-1)$。默认两层 kernel=3，感受野为5；kernel=1 对照没有词间混合。完成的历史句子允许双向卷积，目标句子不进入 CNN。

复用已有 `PoolingMemory`，以4个可训练查询汇总全部合法词位置：

$$
A_{q,i}=\operatorname{softmax}_i\left(q_q^\top W_Kh_i^{(L)}/\sqrt D\right),\qquad m_q=\sum_i A_{q,i}\operatorname{ValueMLP}(h_i^{(L)}).
$$

输出与 `pooling4` 一致：4个语义槽、均匀先验、空句Mask、0维句法系数。Semantic Reader、Decoder、损失、训练调度及断点机制均复用现有实现。槽是不同查询汇总，不解释为4棵树。

## 对照组与结论边界

| 组别 | 用途 |
| --- | --- |
| `no_memory` | 检查历史记忆本身的价值 |
| `pooling4` | 原有逐词变换与全句汇总 |
| `cnn_k1` | 加入相同层数的逐词非线性，不混合邻居 |
| `cnn_k3` | 小窗口有序特征，默认感受野5 |
| `cnn_k5`（可选） | 扩大局部感受野，默认9 |
| `fixed_tree4`（可选） | 固定树上的递归组合 |
| `forest_scalar4` | 主要森林对照，Reader输入维度与CNN一致 |
| `forest_vector4`（可选） | 森林另带句法系数，单独报告其额外因素 |

CNN的pooling会访问整句，不能把最终记忆称为只依赖5个词。kernel=1的整个编码汇总路径对词序置换不变；kernel>1可以编码局部顺序。CNN优于pooling只能支持局部组合有帮助；森林进一步优于CNN，才能支持本任务上复杂结构具有增益。相近的点估计也不构成统计上的等效证明，需要预设实用差异阈值、足够预算和不确定性估计。

各组共用缓存/划分、目标Token预算、有效batch、Decoder结构、学习率和调度。相同seed的Decoder及语义嵌入初始化相同；后续辅助模块/Reader的随机初始化受模块构造顺序影响，并非所有共享模块都逐张量配对。使用多seed并报告逐seed结果。辅助参数量不相同，入口会导出真实参数分组；可调整 `--cnn-hidden-size` 做额外的参数量敏感性对照，不自动声称严格参数匹配。kernel=1与3的参数量也不同。

正式比较至少报告开发集NLL/PPL、逐故事配对差异、多个seed、训练曲线、CE Token/s、峰值显存、参数量及精度策略。先比较同训练Token预算；再补同墙钟预算的效率曲线。历史关闭和跨故事打乱记忆用于检查模型是否使用历史。最终测试集仅在方案固定后评估。

## 入口

在项目根目录用CUDA环境的Python执行，下方 `python` 在本机对应 `E:/anaconda3/envs/py312/python.exe`。

先生成计划和参数表，默认不训练、不下载数据：

```bash
python local_cnn_ablation.py --dry-run --config configs/ablations/seq_cnn.yaml --output-dir runs/cnn_plan
```

正式训练使用自己准备好的完整数据缓存，并显式指定预算。下面示例是主模型、5组、3个seed、每组25M目标Token，会依次训练15次；不是本次已经执行的实验：

```bash
python local_cnn_ablation.py --train --evaluate --config configs/ablations/seq_cnn_main.yaml --cache-dir cache/tinystories --budget-tokens 25000000 --seeds 13 42 2026 --output-dir runs/cnn_main_25m
```

`--evaluate` 在各组完成后用完整开发集进行normal/memory_off/shuffled_memory评估，输出逐故事CSV，可交给现有 `evaluate.py --bootstrap A.csv B.csv` 做逐故事配对bootstrap。多个seed的不确定性应另行汇总，不能只bootstrap单个seed。默认变体为no_memory、pooling4、cnn_k1、cnn_k3、forest_scalar4；`--variants` 可选上表其他组。输出包含 `plan.json`、`parameters.csv`、每组解析后的JSON配置、原训练日志和检查点。已有日志的目录拒绝作为新训练覆盖，单组续训使用原有 `train.py --resume`。

仅测CUDA吞吐（完整forward/backward/clip/AdamW，不保存训练检查点）：

```bash
python local_cnn_ablation.py --benchmark --config configs/ablations/seq_cnn.yaml --seeds 42 --source synthetic --warmup 3 --repeats 10 --output-dir runs/cnn_timing
```

默认输入B=2、H=4、W=24、T=32，各组使用同一个驻留GPU的batch；遵循配置的AMP/Composer/Reader精度策略。`--source cache --cache-dir ...` 可用真实输入替代合成输入。输出每次计时原始值、汇总CSV和环境记录。顺序运行、少量样本仅用于工程检查，正式速度结论应增加重复并控制GPU温度/负载与运行顺序。

直接训练单个CNN：

```bash
python train.py --config configs/ablations/seq_cnn.yaml --cache-dir cache/tinystories --output-dir runs/cnn_single
```

顶层 `Train.ipynb` 也可使用：同时设置 `CONFIG="configs/ablations/seq_cnn.yaml"` 和 `MODE="seq_cnn4"`，其余步骤照常。kernel=1配置为 `configs/ablations/seq_cnn_k1.yaml`，主模型为 `configs/ablations/seq_cnn_main.yaml`。

## 工程验证

`tests/test_seq_cnn.py` 检验局部感受野、左右顺序敏感性、kernel=1不混合邻居、padding隔离、空句、全模型梯度、目标因果性和权重恢复；原有全模式测试自动覆盖新增CNN模式。实际CUDA smoke训练/恢复与计时结果记录在同目录工程报告中；这些短跑不用于架构效果结论。

2026-11-03实测：相关模型/训练恢复/数据因果性/精度策略/Reader检查共 **53 passed**。真实小缓存、CNN smoke、有效128 CE Token/update完成256 Token训练，经原 `train.py --resume` 续训至384 Token；优化器step、调度器计数、数据指纹和日志连续性均通过。256 Token检查点完成小缓存全部开发集1041个目标Token的评估。原始记录见 `engineering_validation.json`、`training_entry.log`、`resume.log`、`training_metrics.csv`、`dev_engineering.csv`。这些数值只检查工程流程，未开展正式NLL效果对照。

RTX3070 Laptop、BF16外层AMP、FP32 IEEE森林Composer、BF16 Reader，B2/H4/W24/T32，预热3次、计时10次的完整更新中位数如下。包含forward/backward/梯度裁剪/AdamW；输入驻留GPU，不含数据读取、评估或检查点I/O。

| 组别 | 参数量 | 中位更新耗时(ms) | 目标Token/s | 峰值已分配显存(GiB) |
| --- | ---: | ---: | ---: | ---: |
| no_memory | 10,093,440 | 23.73 | 2697 | 0.208 |
| pooling4 | 15,781,794 | 54.56 | 1173 | 0.295 |
| cnn_k1 | 15,914,402 | 47.87 | 1337 | 0.297 |
| cnn_k3 | 16,176,546 | 59.38 | 1078 | 0.301 |
| forest_scalar4 | 22,385,795 | 314.48 | 204 | 0.410 |

该形状下CNN k3比标量森林约快5.30倍，但参数和历史编码计算也更少，不能把差异全部归因于CKY。各组顺序测量，CNN k1甚至快于更简单的pooling，说明短测的波动足以影响相近组别的排序；正式性能比较须增加重复、交替运行并监控设备状态。原始10次耗时及配置见 `engineering_benchmark/`。
