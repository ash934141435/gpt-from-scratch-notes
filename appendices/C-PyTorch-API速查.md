# 附录 C：本课 PyTorch API 速查

只收录课程真正使用的 API。每项先说明作用，再列最常见错误。

## 张量创建与检查

### `torch.tensor(data, dtype=...)`

把 Python 数据转为 tensor。

```python
data = torch.tensor([1, 2, 3], dtype=torch.long)
```

Embedding 的索引必须是整数类型，通常使用 `torch.long`。

### `torch.zeros`、`torch.ones`、`torch.randn`

```python
torch.zeros(B, T, C)   # 全 0
torch.ones(T, T)       # 全 1
torch.randn(B, T, C)   # 标准正态随机数
```

`randn` 产生浮点数；不能直接把它当 embedding 索引。

### `.shape`、`.dtype`、`.device`

```python
print(x.shape, x.dtype, x.device)
```

调试张量时三者应一起看。shape 正确但 dtype/device 错误仍会失败。

## 索引、切片和批处理

### 切片 `x[b, :t+1]`

Python 切片右端不包含，因此 `:t+1` 才会包含当前位置 t。

### `torch.arange(T, device=...)`

创建 `[0,1,...,T-1]`，本课用于查询 position embedding。

### `torch.stack`

把多个同 shape tensor 沿新维度堆叠：

```python
x = torch.stack([data[i:i+T] for i in starts])  # [B,T]
```

所有元素 shape 必须相同。

### `torch.cat`

沿已有维度拼接：

```python
idx = torch.cat((idx, idx_next), dim=1)       # 扩展时间
out = torch.cat([head(x) for head in heads], dim=-1)  # 拼通道
```

除拼接维外，其他维度必须相同。

## Shape 操作

### `.view(...)` 与 `.reshape(...)`

```python
logits = logits.reshape(B * T, V)
targets = targets.reshape(B * T)
```

元素总数必须保持。`view` 对内存连续性要求更严格；`reshape` 必要时可创建副本。两者都不会按业务含义自动重排元素。

### `.transpose(-2, -1)`

交换最后两个维度：

```python
k.transpose(-2, -1)  # [B,T,H] → [B,H,T]
```

负索引从末尾计数。不要误交换 batch 维。

### `.mean(dim)` 与 `.sum(dim, keepdim=True)`

```python
x.mean(dim=0)
weights.sum(dim=1, keepdim=True)
```

被统计维会消失，除非 `keepdim=True`。保留长度为 1 的维度常用于后续广播。

## 概率与采样

### `F.softmax(logits, dim=-1)`

把每行 logits 变为非负且和为 1 的概率：

```python
probabilities = F.softmax(logits, dim=-1)
```

必须明确哪一维是候选类别/源位置。对错误 dim 归一化通常不报错，却改变模型逻辑。

### `F.cross_entropy(logits, targets)`

输入：

```text
logits  [N,V]，浮点数
targets [N]，long，值域 0…V-1
```

它内部包含 log-softmax；不要先手动 softmax 再传入。

### `torch.multinomial(probabilities, num_samples=1)`

按每行概率随机采样类别 ID：

```python
idx_next = torch.multinomial(probabilities, 1)  # [B,1]
```

它不是取最大值；固定随机种子才能更容易复现实验。

## Mask 与矩阵

### `torch.tril`

保留主对角线和左下方：

```python
tril = torch.tril(torch.ones(T, T))
```

主对角线为 1 表示 token 可以读取自己。

### `.masked_fill(condition, value)`

条件为 True 的位置被替换：

```python
scores = scores.masked_fill(tril[:T, :T] == 0, float("-inf"))
```

未来分数填 `-inf` 后，softmax 权重为 0；填 0 不能保证被屏蔽。

### `@` / `torch.matmul`

```python
scores = q @ k.transpose(-2, -1)
out = weights @ value
```

支持前导 batch 维。先检查 `[...,M,K] @ [...,K,N]`，再解释 M/N。

### `torch.allclose`

判断浮点 tensor 在误差容限内接近：

```python
assert torch.allclose(loop_result, matrix_result)
```

数值算法通常不应用逐元素 `==` 证明等价。

## 模型组件

### `nn.Embedding(num_embeddings, embedding_dim)`

按整数 ID 查可学习表：

```python
table = nn.Embedding(V, C)
out = table(idx)  # [B,T] → [B,T,C]
```

索引必须在 `[0,V-1]`，且 dtype 为 long。

### `nn.Linear(in_features, out_features, bias=...)`

只变换最后一维：

```python
layer = nn.Linear(C, H, bias=False)
out = layer(x)  # [B,T,C] → [B,T,H]
```

前导 B/T 维不被混合。

### `nn.Sequential`

依次调用子模块：

```python
self.net = nn.Sequential(
    nn.Linear(C, 4 * C),
    nn.ReLU(),
    nn.Linear(4 * C, C),
)
```

它只表达线性流程；残差相加等多分支逻辑应写在 `forward`。

### `nn.ModuleList`

保存可遍历且会被 PyTorch 注册的子模块：

```python
self.heads = nn.ModuleList([Head(H) for _ in range(nh)])
```

普通 Python list 可能使参数不出现在 `.parameters()`、`.to()` 和 `state_dict` 中。

### `register_buffer(name, tensor)`

```python
self.register_buffer("tril", mask)
```

buffer 会保存并随模型移动设备，但不由优化器更新。适合固定 mask，不适合可学习权重。

### `nn.LayerNorm(C)`

对输入最后 C 维计算均值和方差：

```python
norm = nn.LayerNorm(C)
out = norm(x)  # [B,T,C] → [B,T,C]
```

不要把 T 误写成 normalized shape。

### `nn.Dropout(p)`

训练时以概率 p 置零并放大保留项；评估时为恒等映射。

```python
model.train()  # 开启
model.eval()   # 关闭
```

`torch.no_grad()` 只关闭梯度记录，不会自动关闭 Dropout。

## 训练控制

### `torch.manual_seed(seed)`

固定 PyTorch 随机数序列，便于重复调试；不能保证所有硬件/内核完全逐位一致。

### `optimizer.zero_grad(set_to_none=True)`

清除上一轮梯度。PyTorch 默认会累积 `.grad`，忘记清除会把多轮梯度叠加。

### `loss.backward()`

从标量 loss 反向计算所有参与运算且需要梯度的参数梯度。

### `optimizer.step()`

根据当前 `.grad` 更新参数。正确顺序：

```python
optimizer.zero_grad(set_to_none=True)
logits, loss = model(x, y)
loss.backward()
optimizer.step()
```

### `@torch.no_grad()`

关闭函数内梯度图记录，减少评估与生成的内存开销。它不切换 Dropout/LayerNorm 模式。

### `model.train()` 与 `model.eval()`

切换 Dropout、BatchNorm 等模块行为；不负责打开/关闭 autograd。本课评估通常同时使用 `eval()` 与 `no_grad()`。

### `.to(device)`

移动模型或 tensor：

```python
model = model.to(device)
x, y = x.to(device), y.to(device)
```

只移动模型不会自动移动后来创建的输入。注册 buffer 会随模型一起移动。

## 保存、恢复与候选裁剪

### `torch.save(...)` 与 `torch.load(...)`

checkpoint 通常保存字典，而不只保存模型权重：

```python
torch.save({
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "step": step,
}, "checkpoint.pt")

checkpoint = torch.load("checkpoint.pt", map_location=device, weights_only=True)
model.load_state_dict(checkpoint["model"])
optimizer.load_state_dict(checkpoint["optimizer"])
```

`map_location` 让在其他设备保存的 tensor 映射到当前设备。只恢复模型、不恢复 optimizer 状态，可以推理，但不等于精确继续原训练过程。

### `torch.topk(logits, k)`

返回最后一维中最大的 k 个值及其索引。生成时可用第 k 大值作为阈值，把其余 logits 设为 `-inf`，再执行 softmax 与采样。top-k 不改变模型参数，也不保证生成事实正确。

## 常见报错速查

| 报错或现象 | 优先检查 |
|---|---|
| `mat1 and mat2 shapes cannot be multiplied` | Linear 输入最后一维、QK 转置维 |
| `Expected tensor ... same device` | 模型、x/y、位置索引、mask device |
| Embedding index out of range | token ID、位置 T 是否超过 block_size |
| CrossEntropy target dtype error | targets 是否 `torch.long` |
| `view size is not compatible` | transpose 后是否需 `.reshape()` 或 `.contiguous()` |
| loss 能降但生成偷看答案 | causal mask 是否存在、方向是否正确 |
| eval 结果每次不同 | 是否忘记 `model.eval()`，采样本身是否随机 |

---

[上一附录：Shape 速查](./B-张量与Shape速查.md) · [下一附录：线性代数](./D-线性代数最低必备知识.md) · [返回课程目录](../course/README.md)
