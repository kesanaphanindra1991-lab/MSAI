import abc

import torch

from .ae import PatchAutoEncoder


def load() -> torch.nn.Module:
    from pathlib import Path

    model_name = "BSQPatchAutoEncoder"
    model_path = Path(__file__).parent / f"{model_name}.pth"
    print(f"Loading {model_name} from {model_path}")
    return torch.load(model_path, weights_only=False)


def diff_sign(x: torch.Tensor) -> torch.Tensor:
    """
    A differentiable sign function using the straight-through estimator.
    Returns -1 for negative values and 1 for non-negative values.
    """
    sign = 2 * (x >= 0).float() - 1
    return x + (sign - x).detach()


class Tokenizer(abc.ABC):
    """
    Base class for all tokenizers.
    Implement a specific tokenizer below.
    """

    @abc.abstractmethod
    def encode_index(self, x: torch.Tensor) -> torch.Tensor:
        """
        Tokenize an image tensor of shape (B, H, W, C) into
        an integer tensor of shape (B, h, w) where h * patch_size = H and w * patch_size = W
        """

    @abc.abstractmethod
    def decode_index(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decode a tokenized image into an image tensor.
        """


class BSQ(torch.nn.Module):
    def __init__(self, codebook_bits: int, embedding_dim: int):
        super().__init__()
        self._codebook_bits = codebook_bits
        self.down_proj = torch.nn.Linear(embedding_dim, codebook_bits, bias=False)
        self.up_proj = torch.nn.Linear(codebook_bits, embedding_dim, bias=False)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Implement the BSQ encoder:
        - A linear down-projection into codebook_bits dimensions
        - L2 normalization
        - differentiable sign
        """
        x = self.down_proj(x)
        x = torch.nn.functional.normalize(x, dim=-1)
        return diff_sign(x)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Implement the BSQ decoder:
        - A linear up-projection into embedding_dim should suffice
        """
        return self.up_proj(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    def encode_index(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run BQS and encode the input tensor x into a set of integer tokens
        """
        return self._code_to_index(self.encode(x))

    def decode_index(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decode a set of integer tokens into an image.
        """
        return self.decode(self._index_to_code(x))

    def _code_to_index(self, x: torch.Tensor) -> torch.Tensor:
        x = (x >= 0).int()
        return (x * 2 ** torch.arange(x.size(-1)).to(x.device)).sum(dim=-1)

    def _index_to_code(self, x: torch.Tensor) -> torch.Tensor:
        return 2 * ((x[..., None] & (2 ** torch.arange(self._codebook_bits).to(x.device))) > 0).float() - 1


class BSQPatchAutoEncoder(PatchAutoEncoder, Tokenizer):
    """
    Combine your PatchAutoEncoder with BSQ to form a Tokenizer.

    Hint: The hyper-parameters below should work fine, no need to change them
          Changing the patch-size of codebook-size will complicate later parts of the assignment.
    """

    def __init__(self, patch_size: int = 5, latent_dim: int = 128, codebook_bits: int = 10):
        # Bottleneck of the underlying PatchAutoEncoder is the BSQ embedding_dim (latent_dim),
        # NOT the codebook_bits -- the BSQ module itself does the latent_dim -> codebook_bits
        # down-projection (and back up) internally.
        super().__init__(patch_size=patch_size, latent_dim=latent_dim, bottleneck=latent_dim)
        self.codebook_bits = codebook_bits
        self.bsq = BSQ(codebook_bits, latent_dim)

    def encode_index(self, x: torch.Tensor) -> torch.Tensor:
        # image -> PatchAutoEncoder encode -> BSQ encode_index -> tokens
        # (do NOT route this through a separately-computed `encode` + `_code_to_index`;
        # encode_index handles the code->index conversion for us)
        return self.bsq.encode_index(self.encoder(x))

    def decode_index(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.bsq._index_to_code(x))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.bsq.encode(self.encoder(x))

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.bsq.decode(x))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Return the reconstructed image and a dictionary of additional loss terms you would like to
        minimize (or even just visualize).
        Hint: It can be helpful to monitor the codebook usage with

              cnt = torch.bincount(self.encode_index(x).flatten(), minlength=2**self.codebook_bits)

              and returning

              {
                "cb0": (cnt == 0).float().mean().detach(),
                "cb2": (cnt <= 2).float().mean().detach(),
                ...
              }
        """
        # NOTE: we compute `code` once here and reuse it for both reconstruction and the
        # (detached, monitoring-only) codebook-usage stats below. We never call
        # `_code_to_index` on anything that needs a gradient -- it's an integer-valued,
        # non-differentiable op, and using it inside the differentiable path would silently
        # block gradient flow back through the encoder.
        z = self.encoder(x)
        code = self.bsq.encode(z)
        x_hat = self.decoder(self.bsq.decode(code))

        with torch.no_grad():
            idx = self.bsq._code_to_index(code)
            cnt = torch.bincount(idx.flatten(), minlength=2**self.codebook_bits)
            stats = {
                "cb0": (cnt == 0).float().mean(),
                "cb2": (cnt <= 2).float().mean(),
            }
        return x_hat, stats
