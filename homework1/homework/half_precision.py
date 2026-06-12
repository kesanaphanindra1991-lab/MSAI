from pathlib import Path

import torch

from .bignet import BIGNET_DIM, LayerNorm  # noqa: F401


class HalfLinear(torch.nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
    ) -> None:
        """
        Half-precision Linear Layer.

        Inherits from torch.nn.Linear so the parameter names ("weight", "bias")
        match the original checkpoint -- state_dict loading then "just works",
        even though the dtypes differ (float16 vs float32). PyTorch will cast
        on copy_.

        Weights are stored in float16 and frozen (requires_grad=False), which
        avoids the numerical instability of backprop through fp16 linears and
        keeps backward memory ~zero.
        """
        super().__init__(in_features, out_features, bias)

        # Replace the float32 parameters with float16 versions, frozen.
        self.weight = torch.nn.Parameter(
            self.weight.data.to(torch.float16), requires_grad=False
        )
        if bias:
            self.bias = torch.nn.Parameter(
                self.bias.data.to(torch.float16), requires_grad=False
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input/output are float32 (x.dtype); the matmul itself runs in float16.
        original_dtype = x.dtype
        out = torch.nn.functional.linear(
            x.to(self.weight.dtype), self.weight, self.bias
        )
        return out.to(original_dtype)


class HalfBigNet(torch.nn.Module):
    """
    A BigNet where all weights are in half precision. LayerNorm stays in
    float32 to avoid numerical instability.
    """

    class Block(torch.nn.Module):
        def __init__(self, channels: int):
            super().__init__()
            self.model = torch.nn.Sequential(
                HalfLinear(channels, channels),
                torch.nn.ReLU(),
                HalfLinear(channels, channels),
                torch.nn.ReLU(),
                HalfLinear(channels, channels),
            )

        def forward(self, x: torch.Tensor):
            return self.model(x) + x

    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def load(path: Path | None) -> HalfBigNet:
    # PyTorch can load float32 states into float16 models (param.copy_ casts).
    net = HalfBigNet()
    if path is not None:
        net.load_state_dict(torch.load(path, weights_only=True))
    return net
