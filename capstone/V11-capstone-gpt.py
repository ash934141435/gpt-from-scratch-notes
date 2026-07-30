"""V11：把教学 GPT 补成可训练、评估、保存、恢复和可控生成的结课版本。"""

from __future__ import annotations

import argparse
import math
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "sources/ng-video-lecture/input.txt"


@dataclass(frozen=True)
class Config:
    batch_size: int = 8
    block_size: int = 64
    n_embd: int = 64
    n_head: int = 4
    n_layer: int = 2
    dropout: float = 0.1
    learning_rate: float = 3e-4
    steps: int = 200
    eval_interval: int = 50
    eval_iters: int = 10
    seed: int = 1337


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("请求了 CUDA，但当前 PyTorch 检测不到可用 CUDA 设备")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("请求了 MPS，但当前 PyTorch 检测不到可用 MPS 设备")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float):
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError("n_embd 必须能被 n_head 整除")
        head_size = n_embd // n_head
        self.heads = nn.ModuleList(
            [Head(n_embd, head_size, block_size, dropout) for _ in range(n_head)]
        )
        self.projection = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        joined = torch.cat([head(x) for head in self.heads], dim=-1)
        return self.dropout(self.projection(joined))


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
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float):
        super().__init__()
        self.attention = MultiHeadAttention(n_embd, n_head, block_size, dropout)
        self.feed_forward = FeedForward(n_embd, dropout)
        self.norm1 = nn.LayerNorm(n_embd)
        self.norm2 = nn.LayerNorm(n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm1(x))
        return x + self.feed_forward(self.norm2(x))


class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size: int, config: Config):
        super().__init__()
        self.block_size = config.block_size
        self.token_embedding = nn.Embedding(vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.Sequential(
            *[
                Block(config.n_embd, config.n_head, config.block_size, config.dropout)
                for _ in range(config.n_layer)
            ]
        )
        self.final_norm = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, vocab_size)
        self.apply(self._init_weights)

        # 深度增加时缩小每个残差分支的输出投影，让初始主干更接近恒等路径。
        residual_std = 0.02 / math.sqrt(2 * config.n_layer)
        for block in self.blocks:
            nn.init.normal_(block.attention.projection.weight, mean=0.0, std=residual_std)
            nn.init.normal_(block.feed_forward.net[2].weight, mean=0.0, std=residual_std)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(
        self, indices: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        _, time = indices.shape
        if time > self.block_size:
            raise ValueError(f"输入长度 {time} 超过 block_size={self.block_size}")
        positions = torch.arange(time, device=indices.device)
        x = self.token_embedding(indices) + self.position_embedding(positions)
        logits = self.lm_head(self.final_norm(self.blocks(x)))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        indices: torch.Tensor,
        count: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        stop_token: int | None = None,
    ) -> torch.Tensor:
        if temperature <= 0:
            raise ValueError("temperature 必须大于 0")
        if top_k is not None and top_k < 1:
            raise ValueError("top_k 必须至少为 1")

        was_training = self.training
        self.eval()
        finished = torch.zeros(indices.size(0), dtype=torch.bool, device=indices.device)
        try:
            for _ in range(count):
                logits, _ = self(indices[:, -self.block_size :])
                next_logits = logits[:, -1] / temperature
                if top_k is not None:
                    k = min(top_k, next_logits.size(-1))
                    cutoff = torch.topk(next_logits, k).values[:, [-1]]
                    next_logits = next_logits.masked_fill(next_logits < cutoff, float("-inf"))
                probabilities = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probabilities, 1)
                if stop_token is not None:
                    next_token = torch.where(
                        finished[:, None],
                        torch.full_like(next_token, stop_token),
                        next_token,
                    )
                    finished |= next_token[:, 0] == stop_token
                indices = torch.cat((indices, next_token), dim=1)
                if bool(finished.all()):
                    break
        finally:
            self.train(was_training)
        return indices


def load_data() -> tuple[torch.Tensor, dict[str, int], dict[int, str]]:
    text = DATA_FILE.read_text(encoding="utf-8")
    chars = sorted(set(text))
    stoi = {char: index for index, char in enumerate(chars)}
    itos = {index: char for char, index in stoi.items()}
    encoded = torch.tensor([stoi[char] for char in text], dtype=torch.long)
    return encoded, stoi, itos


def get_batch(
    data: torch.Tensor,
    config: Config,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    starts = torch.randint(
        len(data) - config.block_size,
        (config.batch_size,),
        generator=generator,
    )
    x = torch.stack([data[start : start + config.block_size] for start in starts])
    y = torch.stack([data[start + 1 : start + config.block_size + 1] for start in starts])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(
    model: GPTLanguageModel,
    datasets: dict[str, torch.Tensor],
    config: Config,
    device: torch.device,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    result: dict[str, float] = {}
    for split_index, (name, data) in enumerate(datasets.items()):
        # 每次评估都重建相同随机序列，使不同训练阶段使用相同的一组窗口。
        generator = torch.Generator().manual_seed(config.seed + 10_000 + split_index)
        losses = []
        for _ in range(config.eval_iters):
            _, loss = model(*get_batch(data, config, device, generator))
            assert loss is not None
            losses.append(loss.detach().cpu())
        result[name] = torch.stack(losses).mean().item()
    model.train(was_training)
    return result


def save_checkpoint(
    path: Path,
    model: GPTLanguageModel,
    optimizer: torch.optim.Optimizer,
    config: Config,
    step: int,
    stoi: dict[str, int],
    train_generator: torch.Generator,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": asdict(config),
        "step": step,
        "stoi": stoi,
        "torch_rng_state": torch.get_rng_state(),
        "train_generator_state": train_generator.get_state(),
    }
    device = next(model.parameters()).device
    if device.type == "cuda":
        payload["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    if device.type == "mps" and hasattr(torch.mps, "get_rng_state"):
        payload["mps_rng_state"] = torch.mps.get_rng_state()
    torch.save(payload, path)


def load_checkpoint(
    path: Path,
    model: GPTLanguageModel,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    train_generator: torch.Generator,
) -> int:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
    train_generator.set_state(checkpoint["train_generator_state"].cpu())
    if device.type == "cuda" and "cuda_rng_state_all" in checkpoint:
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state_all"])
    if device.type == "mps" and "mps_rng_state" in checkpoint and hasattr(torch.mps, "set_rng_state"):
        torch.mps.set_rng_state(checkpoint["mps_rng_state"])
    return int(checkpoint["step"])


def encode_prompt(prompt: str, stoi: dict[str, int], device: torch.device) -> torch.Tensor:
    if not prompt:
        raise ValueError("prompt 不能为空；可使用换行符作为最小起始 token")
    unknown = sorted(set(prompt) - set(stoi))
    if unknown:
        raise ValueError(f"字符级词表不包含这些字符：{unknown!r}")
    return torch.tensor([[stoi[char] for char in prompt]], dtype=torch.long, device=device)


def run(
    config: Config,
    checkpoint_path: Path,
    requested_device: str,
    resume: bool,
    prompt: str,
    generate_count: int,
    temperature: float,
    top_k: int | None,
) -> None:
    torch.manual_seed(config.seed)
    device = choose_device(requested_device)
    data, stoi, itos = load_data()
    split = int(0.9 * len(data))
    datasets = {"train": data[:split], "val": data[split:]}

    model = GPTLanguageModel(len(stoi), config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    train_generator = torch.Generator().manual_seed(config.seed)
    start_step = 0
    if resume:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"找不到要恢复的 checkpoint：{checkpoint_path}")
        start_step = load_checkpoint(
            checkpoint_path, model, optimizer, device, train_generator
        )
        if config.steps < start_step:
            raise ValueError(
                f"目标 steps={config.steps} 小于 checkpoint step={start_step}；"
                "请把 --steps 设为不小于已有进度的总步数"
            )

    print(f"运行设备={device}，参数量={sum(p.numel() for p in model.parameters()):,}")
    for step in range(start_step, config.steps):
        if step % config.eval_interval == 0 or step == config.steps - 1:
            metrics = estimate_loss(model, datasets, config, device)
            perplexity = math.exp(metrics["val"])
            print(
                f"训练步数 {step:4d}：训练损失={metrics['train']:.4f}，"
                f"验证损失={metrics['val']:.4f}，验证集困惑度={perplexity:.2f}"
            )
            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                config,
                step,
                stoi,
                train_generator,
            )

        _, loss = model(*get_batch(datasets["train"], config, device, train_generator))
        assert loss is not None
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        config,
        config.steps,
        stoi,
        train_generator,
    )

    # 用新模型实例加载刚保存的文件，防止“只会保存、不能恢复”的假闭环。
    restored = GPTLanguageModel(len(stoi), config).to(device)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=config.learning_rate)
    restored_generator = torch.Generator()
    restored_step = load_checkpoint(
        checkpoint_path, restored, restored_optimizer, device, restored_generator
    )
    context = encode_prompt(prompt, stoi, device)
    generated = restored.generate(
        context,
        count=generate_count,
        temperature=temperature,
        top_k=top_k,
    )[0]
    sample = "".join(itos[index] for index in generated.tolist())
    assert restored_step == config.steps
    assert torch.isfinite(next(restored.parameters())).all()
    print(f"检查点={checkpoint_path}（训练步数={restored_step}）")
    print(sample)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--eval-iters", type=int, default=10)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "capstone/checkpoints/v11.pt")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prompt", default="\n")
    parser.add_argument("--generate", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--smoke-test", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.steps < 1 or args.eval_interval < 1 or args.eval_iters < 1:
        raise ValueError("steps、eval-interval 和 eval-iters 必须为正整数")
    if args.smoke_test:
        config = Config(
            batch_size=2,
            block_size=16,
            n_embd=32,
            n_head=2,
            n_layer=1,
            dropout=0.0,
            steps=2,
            eval_interval=1,
            eval_iters=2,
        )
        with tempfile.TemporaryDirectory() as directory:
            run(
                config,
                Path(directory) / "v11-smoke.pt",
                "cpu",
                False,
                "\n",
                8,
                1.0,
                10,
            )
        return

    config = Config(
        steps=args.steps,
        eval_interval=args.eval_interval,
        eval_iters=args.eval_iters,
    )
    run(
        config,
        args.checkpoint,
        args.device,
        args.resume,
        args.prompt,
        args.generate,
        args.temperature,
        args.top_k,
    )


if __name__ == "__main__":
    main()
