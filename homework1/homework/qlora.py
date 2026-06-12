from pathlib import Path

import torch

from .bignet import BIGNET_DIM, LayerNorm  # noqa: F401
from .low_precision import Linear4Bit


class QLoRALinear(Linear4Bit):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        lora_dim: int,
        group_size: int = 16,
        bias: bool = True,
    ) -> None:
        """
        QLoRA: a 4-bit quantized base (frozen) + a float32 low-rank adapter.

        Just like LoRA, but with Linear4Bit as the base instead of HalfLinear.
        """
        super().__init__(in_features, out_features, bias, group_size)
        self.requires_grad_(False)  # freeze everything from the parent

        # Trainable LoRA adapters in float32.
        self.lora_a = torch.nn.Linear(in_features, lora_dim, bias=False)
        self.lora_b = torch.nn.Linear(lora_dim, out_features, bias=False)

        # A: random, B: zero  =>  initial adapter contribution = 0
        torch.nn.init.kaiming_uniform_(self.lora_a.weight, a=5 ** 0.5)
        torch.nn.init.zeros_(self.lora_b.weight)

        # Ensure float32 and trainable.
        self.lora_a.weight.data = self.lora_a.weight.data.to(torch.float32)
        self.lora_b.weight.data = self.lora_b.weight.data.to(torch.float32)
        self.lora_a.weight.requires_grad_(True)
        self.lora_b.weight.requires_grad_(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base path: quantized linear (no grad through it).
        base = super().forward(x)
        # Adapter path: keep float32 for stable training, then cast back.
        lora_out = self.lora_b(self.lora_a(x.to(torch.float32))).to(x.dtype)
        return base + lora_out


class QLoRABigNet(torch.nn.Module):
    class Block(torch.nn.Module):
        def __init__(self, channels, lora_dim, group_size):
            super().__init__()
            self.model = torch.nn.Sequential(
                QLoRALinear(channels, channels, lora_dim, group_size),
                torch.nn.ReLU(),
                QLoRALinear(channels, channels, lora_dim, group_size),
                torch.nn.ReLU(),
                QLoRALinear(channels, channels, lora_dim, group_size),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.model(x) + x

    def __init__(self, lora_dim: int = 32, group_size: int = 16):
        super().__init__()
        self.model = torch.nn.Sequential(
            self.Block(BIGNET_DIM, lora_dim, group_size),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim, group_size),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim, group_size),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim, group_size),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim, group_size),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim, group_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def load(path: Path | None) -> QLoRABigNet:
    net = QLoRABigNet()
    if path is not None:
        net.load_state_dict(torch.load(path, weights_only=True), strict=False)
    return net
