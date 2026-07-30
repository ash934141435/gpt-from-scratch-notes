# V11 选学：训练闭环

V10 是课程的必学代码终点，默认只训练 20 步，用来快速证明结构、梯度和生成路径可以运行。V11 是教材额外增加的选学工程，不属于 M001–M164 的视频时间轴；不完成它也不影响主线结课。

## 完成后你会得到什么

- 自动选择 CUDA、MPS 或 CPU，也可手动指定设备；
- 使用固定评估窗口平均 train/val loss，使不同训练阶段可直接比较；
- 同时报告验证 loss 和 perplexity；
- 保存模型、优化器、训练步数和随机数状态；
- 从 checkpoint 恢复训练，并用新模型实例验证文件确实可加载；
- 使用 prompt、temperature 和 top-k 控制生成；
- 对 Linear、Embedding 和残差输出投影使用明确初始化。

这仍然是字符级 Tiny Shakespeare 教学模型，不是生产训练框架。它没有加入分布式训练、混合精度、KV cache、学习率调度或工业数据管线。

## 第一次运行

先完成[第 00 章](../course/00-学习前准备/README.md)，在项目根目录执行：

```bash
python capstone/V11-capstone-gpt.py
```

默认配置在普通电脑上完成 200 步训练，并把 checkpoint 写入：

```text
capstone/checkpoints/v11.pt
```

程序会定期打印：

```text
step ...: train=..., val=..., val_ppl=...
```

最后会新建一个模型实例、加载 checkpoint，再生成一段文本。这个过程同时验收“训练、保存、恢复、推理”四个阶段。

## 恢复训练

checkpoint 中的 `step` 表示已完成的训练步数。要把先前训练继续到总计 1000 步：

```bash
python capstone/V11-capstone-gpt.py --resume --steps 1000
```

`--steps` 表示目标总步数，不是额外再训练多少步。模型结构参数必须与保存 checkpoint 时一致；本文件当前把结构配置固定在 `Config` 中，避免命令行误改结构后静默加载错误权重。

## 控制生成

```bash
python capstone/V11-capstone-gpt.py \
  --resume --steps 1000 \
  --prompt "ROMEO:" \
  --generate 400 \
  --temperature 0.8 \
  --top-k 30
```

- `temperature < 1`：分布更尖，结果通常更保守；
- `temperature > 1`：分布更平，结果通常更多样；
- `top-k=30`：每一步只保留分数最高的 30 个候选，再在其中采样；
- 字符级 tokenizer 只能接受训练词表已有字符，遇到新字符会明确报错。

## 为什么评估使用固定窗口

V10 在训练前后各随机抽一个验证 batch，适合烟雾测试，但两个数字可能受到片段难度差异影响。V11 每次评估都用同一随机种子重新建立评估窗口，并平均多个 batch，因此阶段比较更稳定。

Perplexity 定义为：

```text
perplexity = exp(cross_entropy_loss)
```

它可以直观理解为模型在当前 tokenization 和数据分布下的平均不确定程度。不同 tokenizer、上下文切法或数据集上的 perplexity 不能直接横向比较。

## 为什么补充初始化

PyTorch 默认初始化足以让小模型运行，但“能运行”不等于“收敛位置理想”。V11 对 Linear 和 Embedding 使用均值 0、标准差 0.02 的正态初始化，并把每层 attention/FFN 残差输出投影的标准差再除以 `√(2×层数)`。

这里的缩放是一种常见 GPT 工程约定，不是 Transformer 数学定义。它的目标是让许多残差分支叠加时，初始主干仍保持较稳定的尺度。想对比影响，可以暂时注释 `residual_std` 对应的四行初始化，保持其他随机种子和配置不变，再比较前几十步 loss。

## 结课验收问题

1. 为什么评估生成器要在每次 `estimate_loss` 中从相同种子重新建立？
2. 为什么 checkpoint 除模型参数外还要保存 optimizer 和训练步数？
3. `--resume --steps 1000` 为什么不是再训练 1000 步？
4. temperature 和 top-k 分别在哪一步改变概率分布？
5. 为什么不同 tokenizer 下的 perplexity 不能直接比较？
6. 当前生成为何仍会重复计算旧 token，KV cache 将解决哪部分开销？

---

[返回课程目录](../course/README.md) · [代码版本索引](../appendices/E-完整代码版本索引.md) · [延伸主题](../appendices/F-延伸主题与参考资料.md)
