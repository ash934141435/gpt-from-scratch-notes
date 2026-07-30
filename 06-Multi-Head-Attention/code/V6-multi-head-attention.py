"""V6：把四个因果注意力头接入最小字符语言模型。"""

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
        key = self.key(x)
        query = self.query(x)
        scores = query @ key.transpose(-2, -1) * key.shape[-1] ** -0.5
        scores = scores.masked_fill(self.tril[:time, :time] == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        return weights @ self.value(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd: int, num_heads: int, block_size: int):
        super().__init__()
        assert n_embd % num_heads == 0
        head_size = n_embd // num_heads
        self.heads = nn.ModuleList(
            [Head(n_embd, head_size, block_size) for _ in range(num_heads)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([head(x) for head in self.heads], dim=-1)


class TinyLanguageModel(nn.Module):
    def __init__(self, vocab_size: int, n_embd: int, block_size: int, num_heads: int):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.attention = MultiHeadAttention(n_embd, num_heads, block_size)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        _, time = indices.shape
        positions = torch.arange(time, device=indices.device)
        x = self.token_embedding(indices) + self.position_embedding(positions)
        return self.lm_head(self.attention(x))

    def generate(self, indices: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        for _ in range(max_new_tokens):
            context = indices[:, -self.block_size :]
            probabilities = F.softmax(self(context)[:, -1], dim=-1)
            next_index = torch.multinomial(probabilities, num_samples=1)
            indices = torch.cat((indices, next_index), dim=1)
        return indices


def demo() -> None:
    batch, time, n_embd = 2, 8, 32
    model = TinyLanguageModel(vocab_size=65, n_embd=n_embd, block_size=time, num_heads=4)
    indices = torch.randint(65, (batch, time))
    embedded = model.token_embedding(indices)
    head_outputs = [head(embedded) for head in model.attention.heads]
    logits = model(indices)
    generated = model.generate(indices[:, :1], max_new_tokens=time + 2)

    assert len(model.attention.heads) == 4
    assert all(output.shape == (batch, time, 8) for output in head_outputs)
    assert model.attention(embedded).shape == (batch, time, n_embd)
    assert logits.shape == (batch, time, 65)
    assert generated.shape == (batch, time + 3)
    assert "tril" not in dict(model.attention.heads[0].named_parameters())
    assert "tril" in dict(model.attention.heads[0].named_buffers())
    print("每个注意力头的形状：", tuple(head_outputs[0].shape))
    print("拼接后的形状：", tuple(model.attention(embedded).shape))
    print("logits 的形状：", tuple(logits.shape))
    print("生成序列长度：", generated.shape[1])


if __name__ == "__main__":
    demo()
