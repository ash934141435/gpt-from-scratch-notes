# V11 选学：训练闭环

V10 是课程的必学代码终点，默认只训练 20 步，用来快速证明结构、梯度和生成路径可以运行。V11 是教材额外增加的选学工程，不属于 M001–M164 的视频时间轴；不完成它也不影响主线结课。

## 主线 V11 与课程的关系

[完整 V11](./V11-capstone-gpt.py)继承 V10 的模型主干，不再增加 Attention 或 Block。它把真实实验外围容易缺失的部分补齐：配置、设备、稳定评估、checkpoint、恢复训练和可控生成。因此应先读完 V10，再把 V11 看成“怎样管理一次训练”，而不是新的模型章节。

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
训练步数 ...：训练损失=...，验证损失=...，验证集困惑度=...
```

最后会新建一个模型实例、加载 checkpoint，再生成一段文本。这个过程同时验收“训练、保存、恢复、推理”四个阶段。

## 运行结果怎么读

先想快速核对整条路径，可以运行不会在仓库留下 checkpoint 的冒烟测试：

```bash
python capstone/V11-capstone-gpt.py --smoke-test
```

参考结构如下，具体临时路径和生成字符会变化：

```text
运行设备=cpu，参数量=17,409
训练步数    0：训练损失=...，验证损失=...，验证集困惑度=...
训练步数    1：训练损失=...，验证损失=...，验证集困惑度=...
检查点=<系统临时目录>/v11-smoke.pt（训练步数=2）
<生成样例>
```

- 第一行确认设备选择和模型构造成功。
- 中间各行使用固定评估窗口；相邻两步很少，冒烟测试只验收流程，不要求 loss 明显下降。
- 检查点行来自一个全新的模型实例成功加载，证明不是只保存未验证。
- 临时目录会在测试结束后自动清理。正常运行不使用临时目录，会写入前述 `capstone/checkpoints/v11.pt`。

## 完整 V11 代码导读

V11 较长，按职责分成八块阅读，不能把 451 行当成一个整体硬啃。

| 代码区块 | 负责什么 | 相比 V10 的变化 |
|---|---|---|
| 导入、`ROOT/DATA_FILE`、`Config` | 集中保存路径和不可变训练配置 | 超参数不再散落为多组全局常量 |
| `choose_device` | 自动选择 CUDA→MPS→CPU，也校验手动请求 | 增加明确设备错误，不静默回退 |
| `Head`、`MultiHeadAttention`、`FeedForward`、`Block` | 保留 V10 的 Decoder-Only 主干 | Multi-Head 的整除条件改成可读异常 |
| `GPTLanguageModel.__init__/_init_weights` | 组装模型并统一初始化 | Linear/Embedding 用标准差 0.02，残差输出随深度缩放 |
| `forward/generate` | 计算 logits/loss，并带 temperature、top-k、停止 token 生成 | 检查窗口长度和参数范围，生成后恢复原训练模式 |
| `load_data/get_batch/estimate_loss` | 读取、切 batch，并在固定窗口上平均 train/val loss | 训练抽样和评估抽样使用各自的 `Generator` |
| `save_checkpoint/load_checkpoint/encode_prompt` | 保存与恢复完整状态，并校验 prompt 字符 | 不只保存模型权重，还保存 optimizer、step 和随机状态 |
| `run/parse_args/main` | 串联训练、周期评估、恢复、再次加载和命令行入口 | 同时支持正常配置与自动清理文件的 smoke 配置 |

模型主干中有三处值得单独说明：

- `self.apply(self._init_weights)` 会递归访问已注册的 Linear 和 Embedding；随后再覆盖 attention projection 和 FFN 第二个 Linear 的权重，使残差输出使用更小标准差。
- `forward` 在 T 超过 `block_size` 时直接报错；`generate` 则主动裁剪最近窗口，所以总输出可以比窗口长。
- `generate` 先记住 `was_training`，在 `finally` 中恢复原模式。即使采样中途报错，也不会让训练模型意外永久停在 eval 模式。

训练与 checkpoint 部分按一次完整事件链理解：

```text
解析参数 → 建配置/设备/数据/模型/optimizer
        → 可选加载 checkpoint
        → 固定窗口评估 → 随机训练 batch 更新
        → 保存最终 checkpoint
        → 新建模型并加载 → 编码 prompt → 生成
```

`save_checkpoint` 保存 CPU、训练 batch 生成器以及可用设备的随机状态，`load_checkpoint` 以相反顺序恢复并返回 step。`--steps` 因而表示目标总步数：恢复到 200 后传 1000，只执行 200–999。`main()` 对正整数参数先做校验；`--smoke-test` 则用 `TemporaryDirectory`、两步小模型和 CPU 调用同一个 `run()`，所以测试的仍是真实路径，不是另一套假实现。

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
