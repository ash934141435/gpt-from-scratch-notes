"""V7：逐 token FeedForward 的最小可运行快照。"""

import torch
from torch import nn


torch.manual_seed(1337)


class FeedForward(nn.Module):
    def __init__(self, n_embd: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_embd, n_embd), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def demo() -> None:
    batch, time, channels = 2, 4, 8
    layer = FeedForward(channels)
    x = torch.randn(batch, time, channels)
    changed = x.clone()
    changed[:, 0] += 10

    out = layer(x)
    changed_out = layer(changed)
    assert out.shape == x.shape
    assert torch.allclose(out[:, 1:], changed_out[:, 1:])
    assert not torch.allclose(out[:, 0], changed_out[:, 0])
    print("输入/输出形状：", tuple(out.shape))
    print("修改第 0 个词元不会影响后续词元的输出：是")


if __name__ == "__main__":
    demo()
