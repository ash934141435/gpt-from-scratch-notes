# 附录 E：V0–V11 完整代码版本索引

本项目保留三条代码轨道，不能混为一谈：

1. **V0–V10 视频主线学习快照**：为普通电脑准备，可直接运行并带最小自检；其中 V4、V5、V7、V8 会刻意隔离单个概念。
2. **作者累计源码**：位于 `sources/ng-video-lecture` 的 Git 历史，每个提交保存视频当时的完整 `v2.py`，用于查看“上一版完整模型如何变成下一版”。
3. **V11 教材结课选学**：不属于原视频时间轴，用一个程序补齐设备、固定评估、checkpoint、恢复训练和可控生成。

学习时先运行视频主线快照理解单一机制，再对照作者累计源码观察它被接回完整模型的位置。V10 是必学终点；只有想继续练工程闭环时才运行 V11。

## 版本总览

| 版本 | 视频阶段 | 核心变化 | 解决的问题 | 可运行文件 |
|---|---:|---|---|---|
| V0 | 07:51 | 读取文本、字符词表、encode/decode | 字符串不能直接作为模型索引 | [V0](../course/02-文本如何变成数字/code/V0-text-vocabulary.py) |
| V1 | 17:46 | train/val 切分、错位 x/y、随机 batch | 没有监督学习样本 | [V1](../course/03-如何制作训练题目/code/V1-data-pipeline.py) |
| V2 | 21:53 | Bigram embedding、cross-entropy、generate | 没有预测器与采样循环 | [V2](../course/04-第一个Bigram模型/code/V2-bigram-model.py) |
| V3 | 34:57 | AdamW、训练循环、train/val 评估 | 参数不更新，无法判断泛化 | [V3](../course/06-模型如何学习/code/V3-trained-bigram.py) |
| V4 | 42:24–58:17 | 循环、矩阵、mask+softmax 三种前缀平均 | token 之间尚无高效因果通信 | [V4](../course/08-矩阵乘法与因果Mask/code/V4-prefix-average-demo.py) |
| V5 | 64:41–79:13 | Q/K/V、因果 mask、缩放单头注意力 | 历史位置只能固定平均 | [V5](../course/10-单头Self-Attention/code/V5-single-head-demo.py) |
| V6 | 79:13–84:45 | Head 模块、buffer、窗口裁剪、多头拼接 | 单头通信视角有限 | [V6](../course/12-Multi-Head-Attention/code/V6-multi-head-attention.py) |
| V7 | 84:45–86:12 | 逐 token Linear+ReLU | 通信后没有独立计算 | [V7](../course/13-FeedForward-Block与残差/code/V7-feed-forward.py) |
| V8 | 87:55–90:30 | `x + F(x)` 残差路径 | 深层网络梯度路径困难 | [V8](../course/13-FeedForward-Block与残差/code/V8-residual-connection.py) |
| V9 | 90:30–97:49 | 投影、4C FFN、pre-norm 完整 Block | Block 不能稳定堆深 | [V9](../course/14-LayerNorm与Pre-Norm/code/V9-transformer-block.py) |
| V10 | 97:49–102:43 | Dropout、参数化层数、完整 loss 与生成 | 容量、泛化与完整训练路径不足 | [V10](../course/15-完整GPT组装/code/V10-complete-gpt.py) |
| V11 | 教材补充 | 固定评估、初始化、checkpoint、temperature、top-k | V10 只适合快速结构验收 | [V11](../capstone/V11-capstone-gpt.py) |

## 作者累计完整阶段

以下提交属于本地仓库 `sources/ng-video-lecture/`，后一项完整包含前一阶段的模型代码：

| 阶段 | Git 提交 | 完整文件 | 本次主要变化 |
|---|---|---|---|
| S0 | `28ef287` | `v2.py` | 建立第二版模型起点 |
| S1 | `8050fde` | `v2.py` | 加入中间 embedding 层 |
| S2 | `28e5fd7` | `v2.py` | 加入位置 embedding |
| S3 | `10024b1` | `v2.py` | 加入单头 Self-Attention |
| S4 | `a6e0bee` | `v2.py` | 加入 Multi-Head Attention |
| S5 | `97dd3f9` | `v2.py` | 加入 FeedForward 计算块 |
| S6 | `5c3a2d2` | `v2.py` | 加入 Block 与残差连接 |
| S7 | `0016836` | `v2.py` | 加入 LayerNorm |
| S8 | `482b15d` | `v2.py` | 加入 Dropout 并整理超参数 |

例如查看加入残差后的完整代码：

```bash
git -C sources/ng-video-lecture show 5c3a2d2:v2.py
```

这些文件忠实保留作者的视频配置，可能执行很久，也可能默认使用 CUDA；它们用于追踪累计演进。可重复快速运行的版本仍以 V0–V10 学习快照为准。

## V0：文本和词表

你应观察：

- 原文件共有 1,115,394 个字符；
- 字符词表为 65；
- `decode(encode(sample)) == sample`。

关键限制：只能编码词表中已有字符；它还是数据表示工具，不是模型。

## V1：训练数据管线

新增：

```text
原始 token [N]
→ train/validation
→ B 个随机起点
→ x,y [B,T]
```

最重要断言：`x[:,1:] == y[:,:-1]`。它证明 y 是原文本向右错位一项，而不是 token ID 加一。

## V2：未训练 Bigram

`Embedding(V,V)` 让每个当前字符直接查出 V 个下字符 logits。它能计算 loss 和生成，但只使用当前一个 token，且初始化随机。

运行时应看到：

```text
training logits: (B*T,V)
initial loss: 约 4–5
generated shape: (1,21)
```

## V3：训练与评估 Bigram

加入标准五步：取 batch、forward、清梯度、backward、step，并分别平均 train/val loss。

本地已验证的快速配置中，初始化验证 loss 约 4.7，训练后约 2.6；具体数字受设备与随机性影响。

## V4：三种等价前缀平均

三条路径：

1. 双循环切片 `.mean(0)`；
2. 归一化下三角矩阵乘 x；
3. 零分数 → `-inf` mask → softmax → 乘 x。

`allclose` 同时证明三者输出相同。第三种形式为 Q/K 数据依赖分数预留接口。

## V5：单头因果 Self-Attention

关键 shape：

```text
q,k,v: [B,T,H]
scores/weights: [B,T,T]
out: [B,T,H]
```

自检检查每行权重和为 1、右上三角为 0，并包含 `H**-0.5` 缩放。

## V6：多头与生成窗口

四个 8 维头从同一 `[B,T,32]` 输入分别聚合，再拼成 `[B,T,32]`。代码还验证：

- `tril` 属于 named buffers 而非 parameters；
- 生成长度可超过 block size；
- 每次模型输入只保留最近窗口。

## V7：FeedForward 隔离实验

修改 token 0 后，其他时间位置的 FFN 输出保持不变。这直接证明 Linear/ReLU 只做逐 token 计算，不承担通信。

V7 隔离单一概念，不是累计完整语言模型；对应的作者完整阶段是上表 S5。

## V8：残差前向与梯度实验

把分支权重初始化为 0：

```text
out = x + 0 = x
d(out.sum)/dx = 1
```

它用实际 autograd 证明 residual identity path，不依赖只背公式。

V8 同样是隔离实验；包含当时全部模型代码的作者阶段是上表 S6。

## V9：完整 Pre-Norm Block

```python
x = x + attention(norm1(x))
x = x + feed_forward(norm2(x))
```

包含多头 output projection 和 `C→4C→C` FFN。自检验证 shape 保持 `[B,T,C]`，并能产生有限输入梯度。

## V10：完整 Decoder-Only GPT

包含：

- token + position embedding；
- N 个 pre-norm Block；
- attention/FFN Dropout；
- final LayerNorm + lm_head；
- cross-entropy、AdamW 训练与生成。

默认使用快速验收配置，而非视频 A100 配置：

```text
batch=8, block=32, C=64, heads=4, layers=2, steps=20
```

文件顶部保留视频配置说明：`batch=64、block=256、C=384、heads=6、layers=6、dropout=0.2`。

## V11：选学训练闭环

V11 不再增加 Transformer 子层，而是补齐一次可复查实验需要的外围能力：

- 自动或手动选择 CUDA、MPS、CPU；
- 用固定窗口平均 train/val loss，并报告 perplexity；
- 明确初始化 Linear、Embedding 和残差输出投影；
- 保存模型、optimizer、step 与随机数状态；
- 从 checkpoint 恢复训练，并用新实例验证可以加载；
- 用 prompt、temperature 和 top-k 控制采样。

完整说明见 [V11 结课训练闭环](../capstone/README.md)。它仍是字符级教学模型，不等同于生产训练框架。

## 推荐运行顺序

在项目根目录执行：

```bash
python course/02-文本如何变成数字/code/V0-text-vocabulary.py
python course/03-如何制作训练题目/code/V1-data-pipeline.py
python course/04-第一个Bigram模型/code/V2-bigram-model.py
python course/06-模型如何学习/code/V3-trained-bigram.py
python course/08-矩阵乘法与因果Mask/code/V4-prefix-average-demo.py
python course/10-单头Self-Attention/code/V5-single-head-demo.py
python course/12-Multi-Head-Attention/code/V6-multi-head-attention.py
python course/13-FeedForward-Block与残差/code/V7-feed-forward.py
python course/13-FeedForward-Block与残差/code/V8-residual-connection.py
python course/14-LayerNorm与Pre-Norm/code/V9-transformer-block.py
python course/15-完整GPT组装/code/V10-complete-gpt.py
```

到这里主线已经完成。选学 V11 时再执行：

```bash
python capstone/V11-capstone-gpt.py
```

若系统 Python 没有 PyTorch，请先在你自己的虚拟环境安装与设备匹配的 PyTorch；不要为了运行本教材盲目更换系统 Python。

## 版本验收问题

每运行一个版本，都回答：

1. 输入和输出 shape 是什么？
2. 它解决上一版哪个明确限制？
3. 新参数在哪里，是否真的被优化器看见？
4. 删除本次关键行会报错还是静默产生逻辑错误？
5. 当前版本仍不能做什么？

答案说不清时，不要急着运行下一版；回到对应章节的“修改前后对比”和“删除或改错会怎样”。

---

[上一附录：线性代数](./D-线性代数最低必备知识.md) · [下一附录：延伸主题与参考资料](./F-延伸主题与参考资料.md) · [返回课程目录](../course/README.md)
