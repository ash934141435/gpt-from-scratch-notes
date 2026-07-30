# GPT 从零构建：零机器学习基础学习路线

这是一套按学习依赖重写的 18 章中文教材。你只需要会 Python 的变量、列表、循环和函数；类、PyTorch、张量、矩阵、概率、梯度和机器学习都会在第一次需要时解释。

> 完成主线后，你会亲手运行一个小型字符级 decoder-only GPT。它会模仿莎士比亚文本的局部格式，但不是 ChatGPT，也不会获得可靠的知识、推理或对话能力。

## 怎么使用本教材

每章按同一节奏推进：先看具体数字和日常类比，再运行最小代码，随后核对 Shape，最后只学习当前问题需要的公式。视频截图不再单独堆放，而是嵌入对应概念、Shape、设计原因或运行结果；流程和结构优先用 Mermaid、表格与矩阵重画。不要预读术语表；遇到忘记的内容再查附录。

每章完成三层练习：

1. 先照着“完整示范”运行；
2. 再做“填空模仿”；
3. 最后在不看答案的情况下完成“独立小任务”。

只有达到“过关标准”再进入下一章。“暂时不用懂什么”不是遗漏，而是明确推迟的内容。

## 主线进度

总学习时间约 **35.5–50.5 小时**。时间包含阅读、运行、改错和练习，不包含 V11 的长时间训练。

| 章 | 内容 | 视频 / 微段 | 预计时间 | 完成标志 |
|---:|---|---|---:|---|
| 00 | [学习前准备](./00-学习前准备/README.md) | 无视频 | 3–5 小时 | 会运行文件、读 Shape 和常见报错 |
| 01 | [我们究竟要构建什么](./01-我们究竟要构建什么/README.md) | 00:00–07:29 / M001–M011 | 1 小时 | 说清课程边界 |
| 02 | [文本如何变成数字](./02-文本如何变成数字/README.md) | 07:29–12:45 / M012–M017 | 1.5–2 小时 | 编码解码可往返 |
| 03 | [如何制作训练题目](./03-如何制作训练题目/README.md) | 12:45–22:16 / M018–M029 | 2–3 小时 | 手工构造 `[B,T]` |
| 04 | [第一个 Bigram 模型](./04-第一个Bigram模型/README.md) | 22:16–28:37 / M030–M039 | 2–3 小时 | 解释输入、logits、目标与 loss |
| 05 | [逐 token 生成](./05-逐token生成/README.md) | 28:37–34:57 / M040–M047 | 1.5–2 小时 | 画出一次生成的 Shape |
| 06 | [模型如何学习](./06-模型如何学习/README.md) | 34:57–42:24 / M048–M055 | 2–3 小时 | 完成一次参数更新 |
| 07 | [为什么 token 需要交流](./07-token为什么需要交流/README.md) | 42:24–47:03 / M056–M061 | 1–1.5 小时 | 手算前缀平均 |
| 08 | [矩阵乘法与因果 Mask](./08-矩阵乘法与因果Mask/README.md) | 47:03–58:17 / M062–M076 | 3–4 小时 | 验证三种前缀平均等价 |
| 09 | [Embedding 与位置](./09-Embedding与位置/README.md) | 58:17–64:41 / M077–M085 | 1.5–2 小时 | 画出 ID 到 logits 的路线 |
| 10 | [单头 Self-Attention](./10-单头Self-Attention/README.md) | 64:41–71:08 / M086–M093 | 3–4 小时 | 分别解释 Q、K、V |
| 11 | [Attention 的规则和缩放](./11-Attention规则与缩放/README.md) | 71:08–79:13 / M094–M101 | 2–3 小时 | 区分 mask、位置与内容匹配 |
| 12 | [Multi-Head Attention](./12-Multi-Head-Attention/README.md) | 79:13–84:45 / M102–M108 | 1.5–2 小时 | 解释并行与拼接 |
| 13 | [FeedForward、Block 与残差](./13-FeedForward-Block与残差/README.md) | 84:45–92:44 / M109–M120 | 3–4 小时 | 区分交流与逐 token 计算 |
| 14 | [LayerNorm 与 Pre-Norm](./14-LayerNorm与Pre-Norm/README.md) | 92:44–97:49 / M121–M128 | 2–3 小时 | 指出归一化维度 |
| 15 | [完整 GPT 的组装与查漏补缺](./15-完整GPT组装/README.md) | 97:49–1:42:43 / M129–M136 | 2–3 小时 | 从输入完整讲到生成 |
| 16 | [Decoder-Only、原始 Transformer 与 nanoGPT](./16-Decoder-Only与nanoGPT/README.md) | 1:42:43–1:49:00 / M137–M148 | 1.5–2 小时 | 映射课程模型与原架构 |
| 17 | [从预训练到 ChatGPT 式助手](./17-从预训练到ChatGPT/README.md) | 1:49:00–1:56:20 / M149–M164 | 2–3 小时 | 说明助手还增加了什么 |

## 代码主线

`V0 → V1 → … → V10` 是必学主线。V 编号表示“可运行代码里程碑”，章节编号表示“学习步骤”，两者不是一一对应：一个版本可以由相邻两章共同讲解，没有新 V 版本的章节则使用章内小实验隔离概念。

每个版本放在首次要求运行它的章节中。对应 README 统一提供“主线 Vx 与本章的关系”“运行结果怎么读”“完整 Vx 代码导读”；复用同一版本的章节会明确指出本章只关注哪个代码区块。

| 新版章节 | 代码版本 | 说明 |
|---|---|---|
| [02](./02-文本如何变成数字/README.md) | [V0](./02-文本如何变成数字/code/V0-text-vocabulary.py) | 文本、词表和编码/解码 |
| [03](./03-如何制作训练题目/README.md) | [V1](./03-如何制作训练题目/code/V1-data-pipeline.py) | 训练数据与 x/y batch |
| [04](./04-第一个Bigram模型/README.md) | [V2](./04-第一个Bigram模型/code/V2-bigram-model.py) | Bigram 模型；第 05 章继续使用同一版本学习生成 |
| [06](./06-模型如何学习/README.md) | [V3](./06-模型如何学习/code/V3-trained-bigram.py) | 训练循环与评估 |
| [08](./08-矩阵乘法与因果Mask/README.md) | [V4](./08-矩阵乘法与因果Mask/code/V4-prefix-average-demo.py) | 前缀平均、矩阵乘法与因果 Mask |
| [10](./10-单头Self-Attention/README.md) | [V5](./10-单头Self-Attention/code/V5-single-head-demo.py) | 单头因果 Self-Attention |
| [12](./12-Multi-Head-Attention/README.md) | [V6](./12-Multi-Head-Attention/code/V6-multi-head-attention.py) | Multi-Head Attention |
| [13](./13-FeedForward-Block与残差/README.md) | [V7](./13-FeedForward-Block与残差/code/V7-feed-forward.py)、[V8](./13-FeedForward-Block与残差/code/V8-residual-connection.py) | FeedForward 与残差连接 |
| [14](./14-LayerNorm与Pre-Norm/README.md) | [V9](./14-LayerNorm与Pre-Norm/code/V9-transformer-block.py) | LayerNorm 与完整 Pre-Norm Block |
| [15](./15-完整GPT组装/README.md) | [V10](./15-完整GPT组装/code/V10-complete-gpt.py) | 完整 Decoder-Only GPT |

没有列出代码版本的章节会在“本章与代码主线的关系”中说明它承接或预备哪个版本，只需运行章内最小示例，不需要跳到旧目录找文件。V10 是课程终点；[V11 结课工程](../capstone/README.md)中的固定评估、perplexity、checkpoint、temperature 和 top-k 都是选学，不影响结课。

## 卡住时去哪里

- 忘记 Shape 写法：[张量与 Shape 速查](../appendices/B-张量与Shape速查.md)
- 忘记 PyTorch 函数：[PyTorch API 速查](../appendices/C-PyTorch-API速查.md)
- 想补数学：[线性代数最低必备](../appendices/D-线性代数最低必备知识.md)
- 想查代码版本：[V0–V11 代码索引](../appendices/E-完整代码版本索引.md)
- 学完后继续探索：[延伸主题与参考资料](../appendices/F-延伸主题与参考资料.md)

这些附录是查询工具，不是开课前必须背完的前置课。

## 视频与质量说明

第 01–17 章覆盖 M001–M164 和视频 `00:00:00–01:56:20`；第 00 章是教材新增的基础准备章，没有视频证据要求。正文共有 27 张精选视频截图，每张都采用“引入—图片—M/时间图注—分析”的结构，并在章末统一列出“原视频定位与 M 映射”。M 编号到新章节的唯一映射见 [learner-chapter-map.csv](../qa/learner-chapter-map.csv)。字幕与原视频的核验沿用现有人工 QA，新版截图索引记录图片实际所属标题和证据类型。

[从第 00 章开始](./00-学习前准备/README.md)
