"""V9：多头注意力、四倍宽 FFN、残差与 pre-norm 的完整 Block。"""

import torch
from torch import nn
from torch.nn import functional as F


torch.manual_seed(1337)


class Head(nn.Module):
    def __init__(self, n_embd: int, head_size: int, block_size: int):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, time, _ = x.shape
        key, query, value = self.key(x), self.query(x), self.value(x)
        scores = query @ key.transpose(-2, -1) * key.shape[-1] ** -0.5
        scores = scores.masked_fill(self.tril[:time, :time] == 0, float("-inf"))
        return F.softmax(scores, dim=-1) @ value


class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int):
        super().__init__()
        assert n_embd % n_head == 0
        head_size = n_embd // n_head
        self.heads = nn.ModuleList(
            [Head(n_embd, head_size, block_size) for _ in range(n_head)]
        )
        self.projection = nn.Linear(n_embd, n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(torch.cat([head(x) for head in self.heads], dim=-1))


class FeedForward(nn.Module):
    def __init__(self, n_embd: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int):
        super().__init__()
        self.attention = MultiHeadAttention(n_embd, n_head, block_size)
        self.feed_forward = FeedForward(n_embd)
        self.norm1 = nn.LayerNorm(n_embd)
        self.norm2 = nn.LayerNorm(n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm1(x))
        x = x + self.feed_forward(self.norm2(x))
        return x


def demo() -> None:
    batch, time, channels = 2, 8, 32
    block = Block(channels, n_head=4, block_size=time)
    x = torch.randn(batch, time, channels, requires_grad=True)
    out = block(x)
    out.square().mean().backward()

    first_linear = block.feed_forward.net[0]
    last_linear = block.feed_forward.net[2]
    assert out.shape == x.shape
    assert first_linear.out_features == 4 * channels
    assert last_linear.out_features == channels
    assert x.grad is not None and torch.isfinite(x.grad).all()
    print("模块输入/输出形状：", tuple(out.shape))
    print("前馈网络隐藏层宽度：", first_linear.out_features)
    print("输入梯度均为有限值：是")


if __name__ == "__main__":
    demo()
