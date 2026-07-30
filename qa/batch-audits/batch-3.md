# 批次三审计：第 07–09 章

- 范围：M056–M085，循环前缀平均、矩阵/mask、Embedding/位置。
- 数学坡度：先手算一通道，再循环，再归一化下三角，再 `masked_fill+softmax`。
- 数值安全：明确禁止在含 `-inf` 的 masked logits 上使用普通 Dropout。
- Shape：三种前缀平均要求 `allclose`；V、T、C 首次严格分离。
- 冷读：代理无需 Q/K/V 即可完成前缀平均与表示任务。
