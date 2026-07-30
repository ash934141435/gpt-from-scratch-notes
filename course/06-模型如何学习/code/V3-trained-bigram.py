"""V3：训练 Bigram，并分别估计训练集与验证集 loss。"""

from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F


torch.manual_seed(1337)
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
text = (Path(__file__).resolve().parents[3] / "sources/ng-video-lecture/input.txt").read_text(encoding="utf-8")
chars = sorted(set(text))
stoi = {char: index for index, char in enumerate(chars)}
itos = {index: char for index, char in enumerate(chars)}
encode = lambda string: [stoi[char] for char in string]
decode = lambda tokens: "".join(itos[token] for token in tokens)
vocab_size = len(chars)

data = torch.tensor(encode(text), dtype=torch.long)
split_index = int(0.9 * len(data))
train_data, val_data = data[:split_index], data[split_index:]
batch_size, block_size = 32, 8


def get_batch(split: str) -> tuple[torch.Tensor, torch.Tensor]:
    source = train_data if split == "train" else val_data
    starts = torch.randint(len(source) - block_size, (batch_size,))
    x = torch.stack([source[i : i + block_size] for i in starts])
    y = torch.stack([source[i + 1 : i + block_size + 1] for i in starts])
    return x.to(device), y.to(device)


class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        logits = self.token_embedding_table(idx)
        loss = None
        if targets is not None:
            batch, time, channels = logits.shape
            loss = F.cross_entropy(logits.view(batch * time, channels), targets.view(batch * time))
        return logits, loss

    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            probabilities = F.softmax(logits[:, -1, :], dim=-1)
            idx_next = torch.multinomial(probabilities, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


@torch.no_grad()
def estimate_loss(model: nn.Module, eval_iters: int = 20) -> dict[str, float]:
    model.eval()
    result = {}
    for split in ("train", "val"):
        losses = torch.zeros(eval_iters)
        for step in range(eval_iters):
            x, y = get_batch(split)
            _, loss = model(x, y)
            losses[step] = loss.item()
        result[split] = losses.mean().item()
    model.train()
    return result


def demo() -> None:
    model = BigramLanguageModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    before = estimate_loss(model)

    for _ in range(500):
        x, y = get_batch("train")
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    after = estimate_loss(model)
    assert after["train"] < before["train"]
    assert after["val"] < before["val"]

    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    sample = decode(model.generate(context, 80)[0].tolist())
    print("运行设备：", device)
    print("训练前：", {"训练集": round(before["train"], 4), "验证集": round(before["val"], 4)})
    print("训练后：", {"训练集": round(after["train"], 4), "验证集": round(after["val"], 4)})
    print("生成样例：", repr(sample))


if __name__ == "__main__":
    demo()
