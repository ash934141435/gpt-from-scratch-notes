"""V2：未训练的字符级 Bigram 模型、loss 与逐 token 生成。"""

from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F


torch.manual_seed(1337)
text = (Path(__file__).resolve().parents[2] / "sources/ng-video-lecture/input.txt").read_text(encoding="utf-8")
chars = sorted(set(text))
vocab_size = len(chars)
stoi = {char: index for index, char in enumerate(chars)}
itos = {index: char for index, char in enumerate(chars)}
encode = lambda string: [stoi[char] for char in string]
decode = lambda tokens: "".join(itos[token] for token in tokens)

data = torch.tensor(encode(text), dtype=torch.long)
split_index = int(0.9 * len(data))
train_data = data[:split_index]
val_data = data[split_index:]
batch_size = 4
block_size = 8


def get_batch(split: str) -> tuple[torch.Tensor, torch.Tensor]:
    source = train_data if split == "train" else val_data
    starts = torch.randint(len(source) - block_size, (batch_size,))
    x = torch.stack([source[i : i + block_size] for i in starts])
    y = torch.stack([source[i + 1 : i + block_size + 1] for i in starts])
    return x, y


class BigramLanguageModel(nn.Module):
    def __init__(self, vocabulary_size: int):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocabulary_size, vocabulary_size)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        logits = self.token_embedding_table(idx)
        loss = None
        if targets is not None:
            batch, time, channels = logits.shape
            logits = logits.view(batch * time, channels)
            targets = targets.view(batch * time)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits = logits[:, -1, :]
            probabilities = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
        return idx


def demo() -> None:
    model = BigramLanguageModel(vocab_size)
    x, y = get_batch("train")
    flat_logits, loss = model(x, y)
    assert flat_logits.shape == (batch_size * block_size, vocab_size)
    assert loss is not None and torch.isfinite(loss)

    context = torch.zeros((1, 1), dtype=torch.long)
    generated = model.generate(context, max_new_tokens=20)
    assert generated.shape == (1, 21)
    assert generated[0, 0].item() == 0

    print("训练用 logits 的形状：", tuple(flat_logits.shape))
    print("初始损失：", round(loss.item(), 4))
    print("生成结果的形状：", tuple(generated.shape))
    print("未训练模型的样例：", repr(decode(generated[0].tolist())))


if __name__ == "__main__":
    demo()
