# 附录 A：术语表

本表按本课程中的实际用法解释术语。第一次遇到陌生词时先看“一句话”，需要时再回到对应章节。

## 数据与文本

| 术语 | 一句话解释 | 本课中的具体位置 |
|---|---|---|
| corpus / 语料库 | 用来训练语言模型的大量文本集合 | Tiny Shakespeare 的约 111 万字符 |
| token | 模型每一步处理的离散符号 | 本课主要是单个字符；大型 GPT 常用 subword |
| vocabulary / 词表 | 所有合法 token 的集合 | `chars` 共 65 个字符 |
| tokenizer | 文本和 token ID 之间的转换规则 | `stoi/itos` 是最小字符 tokenizer |
| encode | 把文本转换为整数 ID 序列 | `"hi" → [46,47]`，具体数字由词表决定 |
| decode | 把 token ID 序列还原为文本 | 生成后把整数列表拼回字符串 |
| context | 预测当前下一 token 时可读取的历史 token | 最多最近 `block_size` 个 token |
| block size | 单次前向传播允许的最大上下文长度 T | 教学从 8 放大到 256 |
| batch | 一次并行处理的多条独立序列 | B 条序列互不通信 |
| train split | 用于计算梯度和更新参数的数据 | 语料前 90% |
| validation split | 不用于参数更新、用于估计泛化的数据 | 语料后 10% |
| data leakage | 训练时意外读取本不应可见的信息 | 使用验证数据训练或注意力偷看未来 |

## 张量与 PyTorch

| 术语 | 一句话解释 | 容易混淆点 |
|---|---|---|
| tensor / 张量 | 具有统一 dtype 和明确 shape 的多维数字数组 | 不是“神秘数学对象”，可先看作多维数组 |
| scalar / 标量 | 没有维度的单个数 | loss 通常是标量 |
| vector / 向量 | 一维数列 | 单个 token 表示常为 C 维向量 |
| matrix / 矩阵 | 二维数字表 | 注意力权重对每个 batch/head 是 T×T 矩阵 |
| shape | 每个维度包含多少元素 | 数字相同不代表维度语义相同 |
| dimension / axis | 张量的一个方向 | `dim=-1` 是最后一维，不固定等于某个业务含义 |
| dtype | 元素的数据类型 | embedding 索引需要 `torch.long` |
| device | 张量所在计算设备 | 模型、数据、mask 必须位于同一设备 |
| broadcasting / 广播 | 按规则复用缺失或长度为 1 的维度 | `[B,T,C] + [T,C] → [B,T,C]` |
| view / reshape | 在元素总数不变时重新解释 shape | 不会自动改变元素顺序或数据语义 |
| transpose | 交换两个维度 | QK 点积只交换 K 的 T 和 H |
| parameter | 由优化器通过梯度更新的张量 | Linear 权重、embedding 表、LayerNorm gamma/beta |
| buffer | 随模型保存和移动、但不由优化器更新的张量 | 因果下三角 `tril` |
| module | 可持有参数并定义 `forward` 的模型组件 | Head、Block、完整 GPT 都是 `nn.Module` |

## 语言模型与训练

| 术语 | 一句话解释 | 本课中的具体含义 |
|---|---|---|
| language model | 根据已有 token 预测后续 token 的概率模型 | 字符级下一字符预测 |
| autoregressive / 自回归 | 把自己之前生成的结果作为下一步输入 | 从左到右逐 token 生成 |
| logits | softmax 前可正可负、未归一化的类别分数 | shape `[B,T,vocab_size]` |
| probability | 0 到 1 的归一化可能性 | 对 logits 最后一维做 softmax |
| loss | 用一个数衡量当前预测与目标的差距 | 本课用 cross-entropy |
| cross-entropy | 对正确类别概率取负对数后平均 | 正确 token 概率越高，loss 越低 |
| forward pass | 从输入计算 logits/loss 的过程 | `logits, loss = model(x,y)` |
| backward pass | 从 loss 反向计算各参数梯度 | `loss.backward()` |
| gradient / 梯度 | 参数微小变化对 loss 的局部影响 | 优化器据此决定更新方向 |
| optimizer | 根据梯度更新参数的算法 | 本课使用 AdamW |
| learning rate | 每次更新的基本步长 | 太大可能震荡，太小需要更多步骤 |
| iteration / step | 取一个 batch 并更新一次参数 | 不等于完整遍历一次数据集 |
| evaluation | 在不更新参数时估计 train/val loss | 使用多个随机 batch 平均 |
| perplexity / 困惑度 | `exp(cross-entropy)`，表示当前口径下的平均不确定程度 | 不同 tokenizer 或数据集不能直接比较 |
| overfitting / 过拟合 | 训练集继续改善，验证集改善变慢或变差 | train loss 明显低于 val loss |
| regularization / 正则化 | 抑制只适合训练数据的脆弱解 | Dropout 是本课使用的方法 |
| sampling / 采样 | 按概率随机选择下一个 token | `torch.multinomial`，不是总取最大值 |
| checkpoint | 可用于恢复实验的训练状态文件 | 通常含模型、优化器、step 和配置 |
| temperature | 采样前缩放 logits 的参数 | 小于 1 更尖，大于 1 更平 |
| top-k | 只保留分数最高的 k 个候选再采样 | 改变生成策略，不修改模型参数 |

## Embedding 与 Attention

| 术语 | 一句话解释 | 输入 → 输出 |
|---|---|---|
| embedding | 用一个可学习向量表示离散 ID | token ID `[B,T] → [B,T,C]` |
| token embedding | 表示 token 是什么 | 同一字符在各位置查到同一基础向量 |
| position embedding | 表示 token 在哪里 | 位置 `[0…T-1] → [T,C]` |
| attention | 根据数据依赖权重从其他位置聚合信息 | `[B,T,C] → [B,T,H]`（单头） |
| self-attention | Q/K/V 来自同一组 token 表示 | GPT Block 中的因果注意力 |
| cross-attention | Q 与 K/V 来自不同节点集合 | Decoder Q 查询 Encoder K/V |
| query / Q | 当前目标位置“正在寻找什么”的向量 | `[B,T,C] → [B,T,H]` |
| key / K | 每个源位置“可被怎样匹配”的向量 | `[B,T,C] → [B,T,H]` |
| value / V | 被关注后实际传输的内容向量 | 权重最终与 V 相乘 |
| affinity / score | Q 与 K 点积得到的匹配分数 | `[B,T,H]@[B,H,T] → [B,T,T]` |
| scaled attention | 点积分数乘 `1/√H` | 稳定随 head size 增长的方差 |
| causal mask | 禁止当前位置读取未来 token 的规则 | 未来分数填 `-inf`，softmax 后为 0 |
| attention weight | mask 后经 softmax 的信息聚合比例 | 每个 query 对允许源位置的一行权重 |
| attention head | 一套独立的 Q/K/V 投影与权重图 | 可学习一种通信视角 |
| multi-head attention | 多个 Head 并行后沿 channel 拼接 | `nh×[B,T,H] → [B,T,C]` |

## Transformer Block

| 术语 | 一句话解释 | 课程代码 |
|---|---|---|
| FeedForward / FFN | 对每个 token 独立应用的 MLP | `C → 4C → C` |
| MLP | 线性层与非线性组成的小网络 | Transformer 的逐 token 计算分支 |
| ReLU | `max(0,x)` 非线性 | 教学代码使用 |
| GELU | 平滑门控式非线性 | nanoGPT 中使用 |
| residual connection | 输出写成 `x + F(x)` | 为信息和梯度保留恒等主路 |
| skip connection | residual connection 的另一常用名称 | “绕过复杂分支”的路径 |
| projection | 把分支输出映射回残差宽度 | 多头 `C→C`、FFN `4C→C` |
| normalization | 调整一组特征的均值与方差 | 本课最终使用 LayerNorm |
| BatchNorm | 每个特征跨样本统计 | 作为 LayerNorm 的对照 |
| LayerNorm | 每个 token 对自己的 C 个特征统计 | `[B,T,C]` shape 不变 |
| gamma / beta | LayerNorm 的可学习缩放与偏移 | 允许网络恢复有用尺度 |
| pre-norm | 在复杂子层之前做 LayerNorm | `x + F(LN(x))` |
| post-norm | 残差相加后做 LayerNorm | `LN(x + F(x))` |
| Dropout | 训练时随机置零部分中间值 | 推理前要 `model.eval()` |
| Transformer Block | Attention 与 FFN 加残差和 Norm 的重复单元 | 通信 → 计算 |

## GPT、预训练与对齐

| 术语 | 一句话解释 | 与本课的关系 |
|---|---|---|
| Encoder | 双向读取完整输入并产生表示 | GPT 不包含独立 Encoder |
| Decoder | 因果生成目标 token 的模块 | GPT 只保留 Decoder 主干 |
| decoder-only | 只有因果 Decoder Block 的架构 | 课堂最终模型与 GPT 同类 |
| pre-training / 预训练 | 在大量通用文本上做下一 token 训练 | Tiny Shakespeare 是微型示例 |
| document completer | 根据文本分布继续文档的预训练模型 | 不保证稳定回答问题 |
| fine-tuning / 微调 | 从预训练参数出发，用专门数据继续训练 | 塑造任务或助手行为 |
| SFT | 使用人工高质量示范做监督微调 | 问题后训练生成理想回答 |
| preference data | 人类对多个候选回答的相对排序 | 用于训练奖励模型 |
| reward model | 预测回答符合偏好程度的打分模型 | 它不是最终聊天策略 |
| policy | 给定 prompt 产生 token 概率的生成模型 | PPO 调整的聊天模型 |
| PPO | 受约束地提高期望奖励的策略优化方法 | 视频只介绍流程，未实现 |
| alignment / 对齐 | 让模型行为更符合指定人类目标 | SFT、偏好与强化学习可参与 |
| RLHF | 从人类反馈训练奖励信号并优化策略的一类流程 | 视频提供历史概览，不等于所有现代配方 |

---

[返回课程目录](../course/README.md) · [Shape 速查](./B-张量与Shape速查.md)
