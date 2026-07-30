# 全视频时间轴验收

## 结果

- 视频范围：00:00:00–01:56:20。
- 语义微段：164 个，符合计划中的 120–220 个范围。
- 细粒度字幕事件：2,955 个，全部映射到一个微段。
- M086–M093 保持为已确认的 Self-Attention 样章，没有重新编号。
- 时间轴由上一段结束时间自动生成下一段开始时间，因此不存在人工输入造成的缺口或重叠。
- M164 将最后五秒结束语标记为“合并讲解”，不强行扩写无新增知识画面。

## 章节分布

| 章节 | 内容 |
|---|---|
| 01 | ChatGPT 演示、课程目标、tiny Shakespeare 与 nanoGPT 背景 |
| 02 | 字符词表、Tokenizer、tensor、训练验证集、block 与 batch |
| 03 | Bigram、logits、cross entropy 与 generate |
| 04 | AdamW、训练循环、device 与 loss 评估 |
| 05 | 前缀平均、矩阵乘法、单头注意力与 scaled attention |
| 06 | 接入 Head、生成窗口裁剪与 Multi-Head Attention |
| 07 | Block、残差连接与 LayerNorm |
| 08 | Dropout、规模化训练和生成结果 |
| 09 | encoder-decoder、nanoGPT、预训练与对齐 |

## 后续状态字段

`subtitle-coverage.csv` 中：

- `已覆盖并人工复核`：对应微段已对照字幕、相邻滚动文本、画面和正文完成第二轮检查。
- `已归属，待语义复核`：只有时间归属，不能算内容完成。
- 当前 2,955 个字幕事件均为 `已覆盖并人工复核`；具体记录见 `semantic-review.csv`。

该状态区分“结构已归属”和“语义已人工检查”，避免再次把自动时间映射误报为内容完成。
