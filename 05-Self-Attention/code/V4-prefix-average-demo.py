"""V4：用三种等价写法实现因果前缀平均。"""

import torch
from torch.nn import functional as F


torch.manual_seed(1337)
B, T, C = 2, 4, 3
x = torch.randn(B, T, C)

# 写法 1：直接表达定义，清楚但慢。
loop_average = torch.zeros_like(x)
for batch in range(B):
    for time in range(T):
        loop_average[batch, time] = x[batch, : time + 1].mean(dim=0)

# 写法 2：归一化下三角矩阵一次完成所有前缀平均。
weights = torch.tril(torch.ones(T, T))
weights = weights / weights.sum(dim=1, keepdim=True)
matrix_average = weights @ x

# 写法 3：mask + softmax，形式可以直接升级为注意力。
scores = torch.zeros(T, T)
scores = scores.masked_fill(torch.tril(torch.ones(T, T)) == 0, float("-inf"))
softmax_weights = F.softmax(scores, dim=-1)
softmax_average = softmax_weights @ x


def demo() -> None:
    assert loop_average.shape == matrix_average.shape == (B, T, C)
    assert torch.allclose(loop_average, matrix_average)
    assert torch.allclose(matrix_average, softmax_average)
    assert torch.allclose(softmax_weights.sum(dim=-1), torch.ones(T))
    assert torch.count_nonzero(torch.triu(softmax_weights, diagonal=1)) == 0
    print("x 的形状：", tuple(x.shape))
    print("权重的形状：", tuple(weights.shape))
    print("输出的形状：", tuple(matrix_average.shape))
    print("因果前缀权重：\n", softmax_weights)


if __name__ == "__main__":
    demo()
