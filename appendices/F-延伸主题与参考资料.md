# 附录 F：延伸主题与参考资料

本附录把“原视频已经完整讲解的内容”和“成为更完整 GPT 工程还要继续学习的内容”分开。18 章主线的代码目标是完成 V10 字符级 decoder-only Transformer；下面的条目是下一步地图，V11 和工业能力均为选学。

## 1. 模型初始化

上游教学仓库特别说明：视频没有深入展开初始化，但初始化会影响收敛速度。PyTorch 默认初始化足以让课堂小模型运行；模型变深后，应进一步检查：

- Linear、Embedding 的初始分布和尺度；
- bias 是否从 0 开始；
- 多个残差分支累加时，输出投影是否需要随层数缩放；
- 初始化后的 logits 是否过尖，初始 loss 是否接近合理随机基线；
- 梯度范数是否有限，深层之间是否相差过大。

[V11 结课工程](../capstone/README.md)给出一种明确而可实验的初始化方案。它是工程约定，不是 attention 公式的一部分，也不是所有 GPT 都必须使用的唯一配置。

## 2. 从字符 tokenizer 到 Subword

主线用字符 tokenizer，是为了让 `stoi/itos` 完全透明。更实际的 tokenizer 练习还应覆盖：

1. 在训练语料上学习 BPE 或其他 subword 规则；
2. 保存 tokenizer 配置，使训练与推理使用完全相同的 ID 映射；
3. 设计 BOS、EOS、PAD、UNK 或角色边界等特殊 token；
4. 验证任意 Unicode 文本如何编码、解码和处理未知输入；
5. 比较 tokenizer 改变后序列长度、词表输出层和训练成本的变化。

Tokenizer 改变后，loss/perplexity 的类别空间也会改变，因此不能把字符模型与 subword 模型的 perplexity 直接当作同一尺度比较。

## 3. 更可靠的训练与评估

V11 已补齐固定评估窗口、平均 train/val loss、perplexity 和 checkpoint。继续走向正式实验时，还应考虑：

- 单独保留最终 test split，只在模型与超参数确定后使用；
- 记录配置、软件版本、设备、随机种子和训练耗时；
- 保存 best validation checkpoint，而不只保存最后一步；
- 加入学习率 warmup/decay、梯度裁剪和必要的梯度累积；
- 根据设备评估混合精度、编译和分布式训练；
- 用吞吐量、显存和生成质量共同评价，而不只看一次 loss。

固定随机种子能减少实验噪声，但不能保证跨 PyTorch 版本、操作系统和 CPU/GPU 得到逐位相同结果。需要严格复现时，应同时固定软件环境并查阅 PyTorch 的确定性设置。

## 4. 更完整的生成

V10 只按完整 softmax 分布采样；V11 增加 temperature 与 top-k。继续扩展时可研究：

- EOS 停止条件和批量序列分别结束；
- top-p、贪心、beam search 等解码策略的目标差异；
- repetition penalty 等启发式方法的副作用；
- KV cache 如何避免每生成一个 token 都重复计算整个历史窗口；
- 长度、吞吐量、首 token 延迟和显存之间的权衡。

这些生成策略不改变模型训练出的参数，但会显著改变用户实际看到的文本分布。

## 5. 与其他 Transformer 实现对照

课堂实现使用 learned absolute position embedding、LayerNorm、ReLU/GELU 和显式 T×T 因果注意力。阅读其他实现时可能遇到 RoPE、RMSNorm、门控 MLP、分组查询注意力、融合注意力等设计。

学习这些名词时先回答三个问题：它替换了课堂骨架中的哪个位置、输入输出 shape 是否变化、它解决的是模型能力、训练稳定性还是运行效率。不要仅凭名称不同就把它理解成完全不同的模型。

## 6. 从预训练到助手

第 17 章的 SFT、奖励模型与 PPO 是带 2022 来源的历史教学流程图，而不是所有系统的固定配方。进一步学习时应把以下问题分开：

- 预训练数据与下一 token 目标；
- 指令示范数据与 SFT loss mask；
- 人类或模型偏好数据；
- 奖励建模、直接偏好优化或强化学习目标；
- 事实性、帮助性、安全性和拒答行为的独立评估；
- 推理系统、工具调用和检索等模型外组件。

知道训练阶段名称不等于具备实现和验收对齐系统的能力。本教程的可执行终点仍是预训练语言模型。

## 7. 主要资料入口

- [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html)：本教材所依据课程系列的作者入口。
- [ng-video-lecture](https://github.com/karpathy/ng-video-lecture)：视频对应的逐步教学代码与提交历史；仓库 README 声明代码采用 MIT License。
- [nanoGPT](https://github.com/karpathy/nanoGPT)：用于对照 checkpoint、配置、设备与更紧凑 GPT 实现的工程仓库。
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)：Transformer 原始论文。
- [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)：第 17 章涉及的 GPT-3 论文。
- [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)：SFT、偏好排序、奖励模型与 PPO 流程的主要论文来源。
- [PyTorch 文档](https://docs.pytorch.org/docs/stable/)与[可复现性说明](https://docs.pytorch.org/docs/stable/notes/randomness.html)：API 行为、设备差异和确定性边界。

## 8. 引用与再发布边界

- 本教材正文是对课程内容的重新组织与讲解，不应删除原作者和原课程入口。
- 本地 `sources/ng-video-lecture` 中的上游代码不随公开仓库发布；其许可仍以上游仓库声明的 MIT License 为准。
- 视频、自动字幕、论文截图和视频裁剪图不因代码采用 MIT 就自动采用 MIT；公开再发布、改编或商业使用前，应分别确认来源、许可和合理引用要求。
- 公开仓库只保留课程正文实际引用的裁剪图，不发布原始帧和候选裁剪；具体边界见仓库顶层 `THIRD_PARTY_NOTICES.md`。
- 本项目没有替作者或维护者选择整套教材的统一开源许可证；正文与原创代码的授权仍应由权利人决定。

---

[上一附录：代码版本索引](./E-完整代码版本索引.md) · [V11 选学工程](../capstone/README.md) · [返回课程目录](../course/README.md)
