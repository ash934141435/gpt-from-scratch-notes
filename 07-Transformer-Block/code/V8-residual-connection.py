"""V8：验证残差支路初始为零时，前向与梯度都有恒等主路。"""

import torch
from torch import nn


torch.manual_seed(1337)


class ResidualLayer(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.branch = nn.Linear(channels, channels, bias=False)
        nn.init.zeros_(self.branch.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.branch(x)


def demo() -> None:
    x = torch.randn(2, 4, 8, requires_grad=True)
    layer = ResidualLayer(8)
    out = layer(x)
    out.sum().backward()

    assert torch.equal(out, x)
    assert torch.equal(x.grad, torch.ones_like(x))
    print("前向传播初始为恒等映射：", "是" if torch.equal(out, x) else "否")
    print("梯度原样传到输入：", "是" if torch.equal(x.grad, torch.ones_like(x)) else "否")


if __name__ == "__main__":
    demo()
