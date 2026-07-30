# 附录 B：张量与 Shape 速查

## 统一符号

| 符号 | 含义 | 常见值 |
|---|---|---:|
| B | batch 中独立序列数量 | 4、32、64 |
| T | 当前序列长度，且 `T ≤ block_size` | 8、32、256 |
| C | 残差主干/embedding 特征宽度 `n_embd` | 32、64、384 |
| H | 单个注意力头的宽度 `head_size` | `C/n_head` |
| nh | 注意力头数 `n_head` | 4、6 |
| V | 词表大小 `vocab_size` | 字符模型为 65 |

字母只是简写。看到 C 时仍要问它是 embedding channel 还是代码临时用来表示词表类别；两者可能数值不同。

## 从文本到训练 Batch

```text
原始文本                         Python str
encode(text)                     list[int]，长度 N
torch.tensor(..., long)          [N]

一个输入窗口 x                   [T]
对应目标 y                       [T]
y[i] = x[i+1]（来自原序列）

stack B 个窗口：
x, y                             [B,T]
```

为什么 y 不是对 x 做数值加一？“错位一位”指原文本索引向右移动，不是 token ID `+1`。

## Embedding 与位置

```text
idx                              [B,T]，dtype=long
token_embedding(idx)             [B,T,C]
arange(T)                        [T]
position_embedding(arange(T))    [T,C]

[B,T,C] + [T,C]                  [B,T,C]
```

广播时 `[T,C]` 被逻辑复用于 B 条序列。各 batch 共用位置向量，但 token 内容不同。

## 单头 Self-Attention

```text
x                                [B,T,C]
key(x), query(x), value(x)       [B,T,H]
k.transpose(-2,-1)               [B,H,T]

q @ kᵀ                           [B,T,T]
causal mask                      [T,T] → 广播到 [B,T,T]
softmax(dim=-1)                  [B,T,T]
weights @ v                      [B,T,H]
```

`[B,T,T]` 的两个 T 角色不同：

- 倒数第二维：发出 query 的目标位置；
- 最后一维：提供 key/value 的源位置。

最小索引语义：`weights[b,i,j]` 表示第 b 条序列的目标位置 i 从源位置 j 取多少信息。

## Multi-Head Attention

教学版：

```text
每个 Head 输出                    [B,T,H]
nh 个 Head 的 Python list         nh × [B,T,H]
cat(dim=-1)                       [B,T,nh*H] = [B,T,C]
output projection                 [B,T,C]
```

高效四维版：

```text
一次 QKV 投影                     [B,T,3C]
切分并重排 q/k/v                 [B,nh,T,H]
q @ kᵀ                            [B,nh,T,T]
weights @ v                       [B,nh,T,H]
transpose + reshape               [B,T,C]
```

必须满足 `C % nh == 0`，否则等宽头无法完整拼回 C。

## FeedForward 与残差

```text
x                                [B,T,C]
Linear(C,4C)                     [B,T,4C]
ReLU/GELU                        [B,T,4C]
Linear(4C,C)                     [B,T,C]

x + FFN(x)                       [B,T,C]
x + MHA(x)                       [B,T,C]
```

Linear 永远只变换最后一维。它不会把 T 个 token 混在一起；跨 T 通信由 Attention 完成。

## LayerNorm

```text
x                                [B,T,C]
LayerNorm(C)                     [B,T,C]
```

统计发生 B×T 次，每次只使用一个 `x[b,t,:]` 的 C 个数：

```text
mean[b,t]                        标量
variance[b,t]                    标量
normalized[b,t,:]                [C]
```

B 与 T 是前导索引，不参与同一次均值/方差计算。

## 完整 GPT 前向数据流

从 token ID 到 logits 的完整路径，其中 Block 为 pre-norm 结构（V10 的实现方式）：

```text
idx [B,T]
 │
 ├─ token_embedding(idx)           [B,T,C] ──┐
 │                                            ├─ 相加 ──► x [B,T,C]
 └─ position_embedding(arange(T))  [T,C] ────┘
                                                │
        ┌─────────────── Block ×N（每层结构相同）───────────────┐
        │               │                                      │
        │     x ────────┼──────────────────────────┐           │
        │               │                          │           │
        │               ▼                          │           │
        │     LayerNorm → MHA → Dropout            │           │
        │               │                          │           │
        │               └───────────► ( + ) ◄──────┘           │
        │                           │   x = x + attn(ln1(x))   │
        │     x ────────────────────┼──────────────────┐       │
        │                           │                  │       │
        │                           ▼                  │       │
        │                 LayerNorm → FFN → Dropout    │       │
        │                           │                  │       │
        │                           └───────► ( + ) ◄──┘       │
        │                                   │                  │
        └───────────────────────────────────┼──────────────────┘
                                            ▼
                                 final LayerNorm [B,T,C]
                                            │
                                            ▼
                                 lm_head → logits [B,T,V]
```

读图要点：

- 主干 shape 从头到尾是 `[B,T,C]`；`( + )` 是残差相加，要求两侧 shape 完全相同。
- LayerNorm 在分支**内部**（pre-norm），主干 x 本身始终不被归一化，梯度可以沿主干直通。
- 跨 token 通信只发生在 MHA 分支内；FFN 和 LayerNorm 都逐 token 独立工作。

## 词表 Logits 与 Loss

```text
Block 最终表示                   [B,T,C]
lm_head: Linear(C,V)             [B,T,V]

训练 cross_entropy 前：
logits.reshape(B*T,V)            [B*T,V]
targets.reshape(B*T)             [B*T]
loss                             []  标量
```

这里 V 是类别数。每个 `[b,t]` 对应一行 V 个 logits 和一个正确 token ID。

## 生成循环

```text
idx                              [B,current_length]
idx_cond = idx[:,-block_size:]   [B,T]，T≤block_size
model(idx_cond)                  [B,T,V]
logits[:,-1,:]                   [B,V]
logits / temperature             [B,V]
top-k mask（可选）               [B,V]
softmax                          [B,V]
multinomial(...,1)               [B,1]
cat(dim=1)                       [B,current_length+1]
```

模型只读取裁剪窗口 `idx_cond`，但完整 `idx` 继续保存全部生成结果。

temperature 和 top-k 都不改变 shape。它们改变的是 `[B,V]` 中哪些候选拥有多大采样概率。

## Cross-Attention

源序列和目标序列长度不必相等：

```text
q from decoder                   [B,T_target,H]
k,v from encoder                [B,T_source,H]
q @ kᵀ                           [B,T_target,T_source]
weights @ v                     [B,T_target,H]
```

## `view`、`reshape`、`transpose`、`cat`、`stack`

| 操作 | Shape 行为 | 例子 |
|---|---|---|
| `view/reshape` | 元素总数不变，重新分组 | `[B,T,V]→[B*T,V]` |
| `transpose(a,b)` | 交换两个已有维度 | `[B,T,H]→[B,H,T]` |
| `cat(...,dim)` | 沿已有维度接长 | 4×`[B,T,8]→[B,T,32]` |
| `stack(...,dim)` | 新增一个维度 | 4×`[B,T,8]→[4,B,T,8]`（视 dim 而定） |
| `squeeze` | 删除长度为 1 的维度 | `[B,1]→[B]` |
| `unsqueeze` | 新增长度为 1 的维度 | `[T,C]→[1,T,C]` |

## 矩阵乘法检查口诀

对最后两个矩阵维：

```text
[..., M, K] @ [..., K, N] → [..., M, N]
```

1. K 必须相等；
2. K 被逐项乘加后消失；
3. M 与 N 保留；
4. 前导维必须相同或可广播；
5. 最后再为每个保留维赋予业务含义。

---

[上一附录：术语表](./A-术语表.md) · [下一附录：PyTorch API](./C-PyTorch-API速查.md) · [返回课程目录](../course/README.md)
