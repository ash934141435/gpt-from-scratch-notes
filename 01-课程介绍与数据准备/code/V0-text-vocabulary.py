"""V0：读取 Tiny Shakespeare，建立字符词表并验证编码往返。"""

from pathlib import Path


TEXT_PATH = Path(__file__).resolve().parents[2] / "sources/ng-video-lecture/input.txt"
text = TEXT_PATH.read_text(encoding="utf-8")
chars = sorted(set(text))
stoi = {char: index for index, char in enumerate(chars)}
itos = {index: char for char, index in stoi.items()}


def encode(string: str) -> list[int]:
    return [stoi[char] for char in string]


def decode(tokens: list[int]) -> str:
    return "".join(itos[token] for token in tokens)


def demo() -> None:
    sample = "First Citizen:\n"
    tokens = encode(sample)
    assert len(text) == 1_115_394
    assert len(chars) == 65
    assert decode(tokens) == sample
    assert sorted(stoi.values()) == list(range(len(chars)))
    print(f"字符数={len(text):,}，词表大小={len(chars)}")
    print("示例词元：", tokens)
    print("编解码还原结果：", repr(decode(tokens)))


if __name__ == "__main__":
    demo()
