from pathlib import Path

import torch

from .bignet import BIGNET_DIM, LayerNorm  # noqa: F401
from .half_precision import HalfLinear


class LoRALinear(HalfLinear):
    lora_a: torch.nn.Module
    lora_b: torch.nn.Module

    def __init__(
        self,
        in_features: int,
        out_features: int,
        lora_dim: int,
        bias: bool = True,
    ) -> None:
        """
        LoRALinear: a HalfLinear base (frozen, fp16) + two low-rank float32
        adapters lora_a and lora_b whose product is added to the base output.

        Standard LoRA init: A ~ Kaiming, B = 0  =>  the adapter is zero at the
        start, so the model output matches the base model exactly until training.
        """
        super().__init__(in_features, out_features, bias)

        # LoRA adapters kept in float32 for stable training.
        self.lora_a = torch.nn.Linear(in_features, lora_dim, bias=False)
        self.lora_b = torch.nn.Linear(lora_dim, out_features, bias=False)

        # A: random, B: zero  =>  initial adapter output = 0
        torch.nn.init.kaiming_uniform_(self.lora_a.weight, a=5 ** 0.5)
        torch.nn.init.zeros_(self.lora_b.weight)

        # Force float32 (HalfLinear's __init__ doesn't touch these, but be explicit).
        self.lora_a.weight.data = self.lora_a.weight.data.to(torch.float32)
        self.lora_b.weight.data = self.lora_b.weight.data.to(torch.float32)

        # Base weights are already frozen (HalfLinear), make sure LoRA is trainable.
        self.lora_a.weight.requires_grad_(True)
        self.lora_b.weight.requires_grad_(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base path (HalfLinear handles the fp16 cast and back to x.dtype).
        base = super().forward(x)
        # LoRA path in float32, then cast back to x.dtype.
        lora_out = self.lora_b(self.lora_a(x.to(torch.float32))).to(x.dtype)
        return base + lora_out


class LoraBigNet(torch.nn.Module):
    class Block(torch.nn.Module):
        def __init__(self, channels: int, lora_dim: int):
            super().__init__()
            self.model = torch.nn.Sequential(
                LoRALinear(channels, channels, lora_dim),
                torch.nn.ReLU(),
                LoRALinear(channels, channels, lora_dim),
                torch.nn.ReLU(),
                LoRALinear(channels, channels, lora_dim),
            )

        def forward(self, x: torch.Tensor):
            return self.model(x) + x

    def __init__(self, lora_dim: int = 32):
        super().__init__()
        self.model = torch.nn.Sequential(
            self.Block(BIGNET_DIM, lora_dim),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def load(path: Path | None) -> LoraBigNet:
    # strict=False because lora_a / lora_b aren't in the original checkpoint.
    net = LoraBigNet()
    if path is not None:
        net.load_state_dict(torch.load(path, weights_only=True), strict=False)
    return net
