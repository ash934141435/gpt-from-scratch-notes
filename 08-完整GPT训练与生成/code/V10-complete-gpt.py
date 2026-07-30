"""V10：完整 decoder-only GPT；默认使用可在普通电脑快速验收的小配置。

所有模块类都通过构造函数显式接收超参数（与 V6 风格一致），
不读取全局常量；全局常量只在 `demo()` 装配模型时使用。
"""

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


torch.manual_seed(1337)

# 视频规模：batch=64, block=256, n_embd=384, n_head=6, n_layer=6, dropout=0.2。
BATCH_SIZE, BLOCK_SIZE = 8, 32
N_EMBD, N_HEAD, N_LAYER, DROPOUT = 64, 4, 2, 0.2
STEPS, LEARNING_RATE = 20, 3e-4


class Head(nn.Module):
    def __init__(self, n_embd: int, head_size: int, block_size: int, dropout: float):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, time, _ = x.shape
        key, query, value = self.key(x), self.query(x), self.value(x)
        weights = query @ key.transpose(-2, -1) * key.shape[-1] ** -0.5
        weights = weights.masked_fill(self.tril[:time, :time] == 0, float("-inf"))
        weights = self.dropout(F.softmax(weights, dim=-1))
        return weights @ value


class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd: int, num_heads: int, block_size: int, dropout: float):
        super().__init__()
        assert n_embd % num_heads == 0
        self.heads = nn.ModuleList(
            [Head(n_embd, n_embd // num_heads, block_size, dropout) for _ in range(num_heads)]
        )
        self.projection = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.projection(torch.cat([head(x) for head in self.heads], dim=-1)))


class FeedForward(nn.Module):
    def __init__(self, n_embd: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd: int, num_heads: int, block_size: int, dropout: float):
        super().__init__()
        self.attention = MultiHeadAttention(n_embd, num_heads, block_size, dropout)
        self.feed_forward = FeedForward(n_embd, dropout)
        self.norm1 = nn.LayerNorm(n_embd)
        self.norm2 = nn.LayerNorm(n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm1(x))
        return x + self.feed_forward(self.norm2(x))


class GPTLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_embd: int,
        block_size: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[Block(n_embd, num_heads, block_size, dropout) for _ in range(num_layers)]
        )
        self.final_norm = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(
        self, indices: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        _, time = indices.shape
        positions = torch.arange(time, device=indices.device)
        x = self.token_embedding(indices) + self.position_embedding(positions)
        logits = self.lm_head(self.final_norm(self.blocks(x)))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, indices: torch.Tensor, count: int) -> torch.Tensor:
        self.eval()
        for _ in range(count):
            logits, _ = self(indices[:, -self.block_size :])
            probabilities = F.softmax(logits[:, -1], dim=-1)
            indices = torch.cat((indices, torch.multinomial(probabilities, 1)), dim=1)
        return indices


def load_data() -> tuple[torch.Tensor, dict[str, int], dict[int, str]]:
    source = Path(__file__).resolve().parents[2] / "sources/ng-video-lecture/input.txt"
    text = source.read_text(encoding="utf-8")
    chars = sorted(set(text))
    stoi = {char: index for index, char in enumerate(chars)}
    itos = {index: char for char, index in stoi.items()}
    return torch.tensor([stoi[char] for char in text], dtype=torch.long), stoi, itos


def get_batch(data: torch.Tensor, batch_size: int, block_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    starts = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[start : start + block_size] for start in starts])
    y = torch.stack([data[start + 1 : start + block_size + 1] for start in starts])
    return x, y


def demo() -> None:
    data, stoi, itos = load_data()
    split = int(0.9 * len(data))
    train_data, validation_data = data[:split], data[split:]
    model = GPTLanguageModel(
        vocab_size=len(stoi),
        n_embd=N_EMBD,
        block_size=BLOCK_SIZE,
        num_heads=N_HEAD,
        num_layers=N_LAYER,
        dropout=DROPOUT,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    model.eval()
    _, initial_loss = model(*get_batch(validation_data, BATCH_SIZE, BLOCK_SIZE))
    model.train()
    for _ in range(STEPS):
        _, loss = model(*get_batch(train_data, BATCH_SIZE, BLOCK_SIZE))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    model.eval()
    _, final_loss = model(*get_batch(validation_data, BATCH_SIZE, BLOCK_SIZE))

    generated = model.generate(torch.zeros((1, 1), dtype=torch.long), count=80)[0]
    sample = "".join(itos[index] for index in generated.tolist())
    assert initial_loss is not None and final_loss is not None
    assert torch.isfinite(final_loss)
    assert generated.numel() == 81
    assert len(model.blocks) == N_LAYER
    print("参数量：", sum(parameter.numel() for parameter in model.parameters()))
    print(f"验证损失变化：{initial_loss.item():.3f} -> {final_loss.item():.3f}")
    print("生成序列长度：", generated.numel())
    print(sample.replace("\n", "\\n"))

    # 回归验证：模块类不依赖全局超参数，换一套配置也能独立构造和前向。
    probe = GPTLanguageModel(
        vocab_size=len(stoi), n_embd=32, block_size=16, num_heads=2, num_layers=1, dropout=0.0
    )
    probe_logits, _ = probe(torch.zeros((1, 4), dtype=torch.long))
    assert probe_logits.shape == (1, 4, len(stoi))
    assert len(probe.blocks) == 1


if __name__ == "__main__":
    demo()
