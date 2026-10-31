# V3 实施前代码审查

现有 Lazy Search 已经把同 width 的所有父 Span 一起做 Top-R/Top-K/探索。此次主要目标是 offset 循环中的四类 stack、pair tensors 的 cat，以及每个 Span 的 Cell/view。N_w=n-w+1，物理槽数由 width/K/R 决定，合法性由每次 forward 的 mask 决定。

候选按 offset、left-slot、right-slot 的 ragged 顺序排列；叶子1槽、width3常为2槽、之后才达到4槽。Candidate ID 按 width/start/row/slot 编号。不能采用固定 K² flat ID。缓存只存几何索引与 ID 布局，key 包含 n/K/R/B*H/device/layout；不存分数、mask、语义或任何可训练状态。

BF16 叶子先相加再转 float；width>=3 至少含一个FP32内部子树，按原类型提升相加。Top-R priority detach 是原离散选择，gather 后的累计 score 不 detach。只对新 noise/eligible buffers 原样 scatter，不覆盖参与 backward 的 score bank。

随机 draw 按 width 依次为 proposal rand_like、beam-priority rand_like、Bernoulli rand_like；形状保持 [B*H,N_w,E/R/K]，故不需要替换探索分布或 RNG replay。同分兼容方案是直接保留旧 compact candidate shape/stride 和 native topk，不添加ID排序规则；由专门同分测试决定是否 Go。

V3-A先批量Gather评分，保留旧Cell/root/pointer适配；V3-B在A通过后移除正常Cell创建并缓存ID解码，现有已batched的selection不再重复重写。C视实际反向性能决定，保持原Lazy闭包与Composer实现共享。逐项完整审查见 feasibility.json。
