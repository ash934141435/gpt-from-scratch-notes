"""V1：把原始文本变成可供语言模型训练的 x/y batch。"""

from pathlib import Path

import torch


TEXT_PATH = Path(__file__).resolve().parents[3] / "sources/ng-video-lecture/input.txt"
text = TEXT_PATH.read_text(encoding="utf-8")

chars = sorted(set(text))
vocab_size = len(chars)
stoi = {char: index for index, char in enumerate(chars)}
itos = {index: char for index, char in enumerate(chars)}


def encode(string: str) -> list[int]:
    return [stoi[char] for char in string]


def decode(tokens: list[int]) -> str:
    return "".join(itos[token] for token in tokens)


data = torch.tensor(encode(text), dtype=torch.long)
split_index = int(0.9 * len(data))
train_data = data[:split_index]
val_data = data[split_index:]

batch_size = 4
block_size = 8
generator = torch.Generator().manual_seed(1337)


def get_batch(split: str) -> tuple[torch.Tensor, torch.Tensor]:
    source = train_data if split == "train" else val_data
    starts = torch.randint(
        len(source) - block_size,
        (batch_size,),
        generator=generator,
    )
    x = torch.stack([source[i : i + block_size] for i in starts])
    y = torch.stack([source[i + 1 : i + block_size + 1] for i in starts])
    return x, y


def demo() -> None:
    x, y = get_batch("train")
    assert vocab_size == 65
    assert decode(encode("hi there")) == "hi there"
    assert data.dtype == torch.long
    assert x.shape == y.shape == (batch_size, block_size)
    assert torch.equal(x[:, 1:], y[:, :-1])
    print(f"字符数={len(text):,}，词表大小={vocab_size}")
    print(f"训练集={len(train_data):,}，验证集={len(val_data):,}")
    print("x 的形状：", tuple(x.shape))
    print("y 的形状：", tuple(y.shape))
    print("第一组 x：", x[0].tolist())
    print("第一组 y：", y[0].tolist())


if __name__ == "__main__":
    demo()
