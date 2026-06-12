from pathlib import Path

import torch

from .bignet import BIGNET_DIM, LayerNorm  # noqa: F401


def block_quantize_3bit(x: torch.Tensor, group_size: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize a 1D tensor to 3-bit precision in groups of `group_size`.

    Each group stores its absolute-max scale (fp16) plus its values as 3-bit
    integers (0..7). The 3-bit values are packed 8-into-3-bytes:
        byte0 = v0 | v1<<3 | (v2 & 0x3)<<6
        byte1 = (v2>>2) | v3<<1 | v4<<4 | (v5 & 0x1)<<7
        byte2 = (v5>>1) | v6<<2 | v7<<5

    Memory cost: 3 bits/value + 16/group_size bits/value for the scale.
    With group_size=32 that's 3.5 bits/value -> ~8.3 MB for the BigNet weights.
    """
    assert x.dim() == 1
    assert x.size(0) % group_size == 0
    assert group_size % 8 == 0

    x = x.view(-1, group_size)
    normalization = x.abs().max(dim=-1, keepdim=True).values
    # Avoid divide-by-zero for all-zero groups.
    norm_safe = normalization.clamp(min=1e-8)
    x_norm = (x + norm_safe) / (2 * norm_safe)
    # 3 bits -> 8 levels (0..7).
    x_quant = (x_norm * 7).round().clamp(0, 7).to(torch.int16)

    num_groups = x.size(0)
    packs_per_group = group_size // 8
    v = x_quant.view(num_groups, packs_per_group, 8)

    byte0 = (v[..., 0] | (v[..., 1] << 3) | ((v[..., 2] & 0x3) << 6)) & 0xFF
    byte1 = ((v[..., 2] >> 2) | (v[..., 3] << 1) | (v[..., 4] << 4) | ((v[..., 5] & 0x1) << 7)) & 0xFF
    byte2 = ((v[..., 5] >> 1) | (v[..., 6] << 2) | (v[..., 7] << 5)) & 0xFF

    packed = torch.stack([byte0, byte1, byte2], dim=-1)             # (num_groups, packs, 3)
    packed = packed.view(num_groups, packs_per_group * 3).to(torch.int8)

    return packed, normalization.to(torch.float16)


def block_dequantize_3bit(
    packed: torch.Tensor, normalization: torch.Tensor, group_size: int
) -> torch.Tensor:
    """Reverse of block_quantize_3bit. Returns a 1D float32 tensor."""
    num_groups = packed.size(0)
    packs_per_group = group_size // 8

    # int8 is signed in PyTorch: cast to int16 and mask with 0xFF to recover
    # the original unsigned byte value.
    p = packed.view(num_groups, packs_per_group, 3).to(torch.int16) & 0xFF
    b0, b1, b2 = p[..., 0], p[..., 1], p[..., 2]

    v = torch.empty(
        num_groups, packs_per_group, 8, dtype=torch.int16, device=packed.device
    )
    v[..., 0] = b0 & 0x7
    v[..., 1] = (b0 >> 3) & 0x7
    v[..., 2] = ((b0 >> 6) & 0x3) | ((b1 & 0x1) << 2)
    v[..., 3] = (b1 >> 1) & 0x7
    v[..., 4] = (b1 >> 4) & 0x7
    v[..., 5] = ((b1 >> 7) & 0x1) | ((b2 & 0x3) << 1)
    v[..., 6] = (b2 >> 2) & 0x7
    v[..., 7] = (b2 >> 5) & 0x7

    v = v.view(num_groups, group_size)
    normalization = normalization.to(torch.float32)
    x_norm = v.to(torch.float32) / 7
    x = (x_norm * 2 * normalization) - normalization
    return x.view(-1)


class Linear3Bit(torch.nn.Module):
    """Same plumbing pattern as Linear4Bit: weight stored as packed 3-bit buffers."""

    def __init__(
        self, in_features: int, out_features: int, bias: bool = True, group_size: int = 32
    ) -> None:
        super().__init__()
        self._shape = (out_features, in_features)
        self._group_size = group_size

        num_groups = (out_features * in_features) // group_size
        bytes_per_group = (group_size // 8) * 3  # 8 values -> 3 bytes

        self.register_buffer(
            "weight_q3",
            torch.zeros(num_groups, bytes_per_group, dtype=torch.int8),
            persistent=False,
        )
        self.register_buffer(
            "weight_norm",
            torch.zeros(num_groups, 1, dtype=torch.float16),
            persistent=False,
        )
        self._register_load_state_dict_pre_hook(
            Linear3Bit._load_state_dict_pre_hook, with_module=True
        )

        self.bias = None
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_features, dtype=torch.float32))

    def _load_state_dict_pre_hook(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        if f"{prefix}weight" in state_dict:
            weight = state_dict[f"{prefix}weight"]
            del state_dict[f"{prefix}weight"]
            weight_flat = weight.detach().to(torch.float32).contiguous().view(-1)
            q3, norm = block_quantize_3bit(weight_flat, self._group_size)
            self.weight_q3.copy_(q3)
            self.weight_norm.copy_(norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            weight = block_dequantize_3bit(
                self.weight_q3, self.weight_norm, self._group_size
            ).view(self._shape)
        return torch.nn.functional.linear(x, weight, self.bias)


class BigNet3Bit(torch.nn.Module):
    """BigNet with all Linear layers replaced by 3-bit quantized versions."""

    class Block(torch.nn.Module):
        def __init__(self, channels: int, group_size: int = 32):
            super().__init__()
            self.model = torch.nn.Sequential(
                Linear3Bit(channels, channels, group_size=group_size),
                torch.nn.ReLU(),
                Linear3Bit(channels, channels, group_size=group_size),
                torch.nn.ReLU(),
                Linear3Bit(channels, channels, group_size=group_size),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.model(x) + x

    def __init__(self, group_size: int = 32):
        super().__init__()
        self.model = torch.nn.Sequential(
            self.Block(BIGNET_DIM, group_size),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, group_size),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, group_size),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, group_size),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, group_size),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, group_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def load(path: Path | None) -> BigNet3Bit:
    net = BigNet3Bit()
    if path is not None:
        net.load_state_dict(torch.load(path, weights_only=True))
    return net