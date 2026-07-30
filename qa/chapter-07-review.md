# 第 7 章验收记录

## 覆盖范围

- 视频：1:24:45–1:37:49
- 微段：M109–M128，共 20 段
- 字幕：该范围全部标记为“已覆盖”
- 复核截图：21 张裁剪图；公开仓库仅保留新版正文实际引用的图片，原始帧不随仓库发布
- 代码快照：V7 FeedForward、V8 残差直通、V9 pre-norm Transformer Block

## 分段修订

粗时间轴的 M109–M128 标题整体落后于实际内容约一段，并把最终 LayerNorm 的前半句切入第 8 章。逐字幕和画面核对后，重新按 FeedForward、Block、残差、输出投影、LayerNorm 的真实演进排列，并将本章终点修正为 1:37:49。

## 人工核对结论

- Attention“通信”与 FeedForward“逐 token 计算”的职责已用 shape 和隔离实验区分。
- Block 的 `n_embd/n_head/head_size` 关系和多层 Sequential 参数独立性已解释。
- 残差的前向恒等与反向系数 1 均有公式和 V8 运行验证。
- attention 输出投影和 `C→4C→C` FFN 的 shape、模型作用和错误后果已覆盖。
- BatchNorm 按列跨样本与 LayerNorm 按 token 特征的统计方向已用二维例子区分。
- running statistics、gamma/beta、pre-norm/post-norm 的差异均已说明。
- V7 验证 FeedForward 不混合 token；V9 验证完整 Block shape 和有限梯度。
- 视频约 2.24、2.08、2.06 的阶段结果与轻微过拟合均有对应解释。

## 已知边界

- V7/V8 是分别隔离概念的最小快照，V9 才包含本章完整 Block。
- Dropout、模型规模化和最终训练配置从第 8 章加入。
