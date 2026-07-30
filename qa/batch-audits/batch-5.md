# 批次五审计：第 13–15 章

- 范围：M109–M136，FFN/残差、LayerNorm、完整 GPT。
- 操作边界：正文明确标出哪些操作混合 token，哪些只沿 C 工作。
- 归一化：统计轴、gamma/beta、pre/post-norm 和两个独立 LayerNorm 均有验收。
- 第 15 章：只新增 Dropout 和规模/资源成本；其他模块逐项指回第 04–14 章。
- 代码终点：V10 必做，V11/checkpoint/perplexity/temperature/top-k 均标为选学。
