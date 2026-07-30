"""V5：带缩放的单头因果自注意力最小演示。"""

import torch
from torch.nn import functional as F


torch.manual_seed(1337)

B, T, C, H = 2, 4, 6, 3
x = torch.randn(B, T, C)

key = torch.nn.Linear(C, H, bias=False)
query = torch.nn.Linear(C, H, bias=False)
value = torch.nn.Linear(C, H, bias=False)

k = key(x)                          # [B, T, H]
q = query(x)                        # [B, T, H]
raw_scores = q @ k.transpose(-2, -1) * H**-0.5  # [B, T, T]

causal_mask = torch.tril(torch.ones(T, T, dtype=torch.bool))
masked_scores = raw_scores.masked_fill(~causal_mask, float("-inf"))
weights = F.softmax(masked_scores, dim=-1)

v = value(x)                        # [B, T, H]
out = weights @ v                   # [B, T, H]


def demo() -> None:
    assert k.shape == q.shape == v.shape == (B, T, H)
    assert raw_scores.shape == weights.shape == (B, T, T)
    assert out.shape == (B, T, H)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(B, T))
    assert torch.count_nonzero(torch.triu(weights, diagonal=1)) == 0
    print("x 的形状：", tuple(x.shape))
    print("q/k/v 的形状：", tuple(q.shape))
    print("注意力权重的形状：", tuple(weights.shape))
    print("输出的形状：", tuple(out.shape))
    print("第一个批次的权重：\n", weights[0])


if __name__ == "__main__":
    demo()
