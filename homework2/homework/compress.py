from pathlib import Path
from typing import cast

import numpy as np
import torch
from PIL import Image

from .autoregressive import AutoregressiveModel as Autoregressive
from .bsq import Tokenizer

# ---------------------------------------------------------------------------
# A small, self-contained arithmetic coder (Witten-Neal-Cleary style, 32-bit
# precision with underflow/carry handling via "pending bits"). We drive its
# per-symbol probability distribution directly from the trained autoregressive
# model's softmax output, so tokens the model is confident about cost very
# few bits and unlikely tokens cost more -- this is what actually gives us
# compression beyond the flat log2(n_tokens) bits/token you'd get by just
# writing out raw token indices.
# ---------------------------------------------------------------------------

_CODE_BITS = 32
_TOP = (1 << _CODE_BITS) - 1
_FIRST_QTR = (_TOP >> 2) + 1
_HALF = 2 * _FIRST_QTR
_THIRD_QTR = 3 * _FIRST_QTR
_TOTAL_PREC = 1 << 16  # fixed integer "budget" each per-token probability distribution is quantized into


def _probs_to_freqs(probs: torch.Tensor, total: int = _TOTAL_PREC) -> list[int]:
    """
    Quantize a probability distribution over the vocabulary into positive integer
    frequency counts summing exactly to `total`. Every symbol gets at least 1 count
    so nothing is ever literally impossible to encode (which would make it infinite-cost).
    """
    probs = probs.detach().to(torch.float64).cpu()
    n = probs.numel()
    probs = probs.clamp(min=1e-9)
    probs = probs / probs.sum()
    freqs = torch.floor(probs * (total - n)).long() + 1
    diff = total - int(freqs.sum().item())
    if diff != 0:
        idx = int(torch.argmax(probs).item())
        freqs[idx] += diff
    return freqs.tolist()


class _BitWriter:
    def __init__(self):
        self.bits: list[int] = []

    def write_bit_plus_pending(self, bit: int, pending: int):
        self.bits.append(bit)
        self.bits.extend([1 - bit] * pending)

    def to_bytes(self) -> bytes:
        bits = self.bits + [0] * ((-len(self.bits)) % 8)
        out = bytearray(len(bits) // 8)
        for i, bit in enumerate(bits):
            if bit:
                out[i // 8] |= 1 << (7 - (i % 8))
        return bytes(out)


class _BitReader:
    def __init__(self, data: bytes):
        self.data = data
        self.nbits = len(data) * 8
        self.pos = 0

    def read_bit(self) -> int:
        if self.pos >= self.nbits:
            self.pos += 1
            return 0
        byte = self.data[self.pos // 8]
        bit = (byte >> (7 - (self.pos % 8))) & 1
        self.pos += 1
        return bit


def _arithmetic_encode(symbols: list[int], freqs_list: list[list[int]], total: int = _TOTAL_PREC) -> bytes:
    writer = _BitWriter()
    low, high, pending = 0, _TOP, 0

    for sym, freqs in zip(symbols, freqs_list):
        cum = [0]
        for f in freqs:
            cum.append(cum[-1] + f)
        cl, ch = cum[sym], cum[sym + 1]

        rng = high - low + 1
        high = low + (rng * ch) // total - 1
        low = low + (rng * cl) // total

        while True:
            if high < _HALF:
                writer.write_bit_plus_pending(0, pending)
                pending = 0
            elif low >= _HALF:
                writer.write_bit_plus_pending(1, pending)
                pending = 0
                low -= _HALF
                high -= _HALF
            elif low >= _FIRST_QTR and high < _THIRD_QTR:
                pending += 1
                low -= _FIRST_QTR
                high -= _FIRST_QTR
            else:
                break
            low *= 2
            high = high * 2 + 1

    pending += 1
    writer.write_bit_plus_pending(0 if low < _FIRST_QTR else 1, pending)
    return writer.to_bytes()


def _arithmetic_decode(data: bytes, num_symbols: int, get_freqs) -> list[int]:
    """
    get_freqs(decoded_so_far: list[int]) -> list[int] frequency table (over the full
    vocabulary) for the NEXT symbol, conditioned on everything decoded so far.
    """
    reader = _BitReader(data)
    low, high = 0, _TOP
    value = 0
    for _ in range(_CODE_BITS):
        value = (value << 1) | reader.read_bit()

    decoded: list[int] = []
    for _ in range(num_symbols):
        freqs = get_freqs(decoded)
        cum = [0]
        for f in freqs:
            cum.append(cum[-1] + f)
        tot = cum[-1]

        rng = high - low + 1
        target = ((value - low + 1) * tot - 1) // rng

        lo, hi = 0, len(freqs) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if cum[mid + 1] > target:
                hi = mid
            else:
                lo = mid + 1
        sym = lo
        cl, ch = cum[sym], cum[sym + 1]

        high = low + (rng * ch) // tot - 1
        low = low + (rng * cl) // tot

        while True:
            if high < _HALF:
                pass
            elif low >= _HALF:
                value -= _HALF
                low -= _HALF
                high -= _HALF
            elif low >= _FIRST_QTR and high < _THIRD_QTR:
                value -= _FIRST_QTR
                low -= _FIRST_QTR
                high -= _FIRST_QTR
            else:
                break
            low *= 2
            high = high * 2 + 1
            value = (value * 2) | reader.read_bit()

        decoded.append(sym)
    return decoded


class Compressor:
    def __init__(self, tokenizer: Tokenizer, autoregressive: Autoregressive):
        super().__init__()
        self.tokenizer = tokenizer
        self.autoregressive = autoregressive

    def compress(self, x: torch.Tensor) -> bytes:
        """
        Compress the image into a torch.uint8 bytes stream (1D tensor).

        Use arithmetic coding.
        """
        device = next(self.autoregressive.parameters()).device
        x = x.to(device)
        if x.dim() == 3:
            x = x[None]

        with torch.no_grad():
            tokens = self.tokenizer.encode_index(x)  # (1, h, w)
            _, h, w = tokens.shape
            seq_len = h * w
            symbols = tokens.view(seq_len).tolist()

            # One forward pass gives us every position's conditional distribution at
            # once, since the model is causal: logits[:, i] only depends on tokens
            # strictly before i, which are exactly the true tokens we're encoding.
            logits, _ = self.autoregressive(tokens)
            n_tokens = logits.shape[-1]
            probs = torch.softmax(logits.view(seq_len, n_tokens).double(), dim=-1)
            freqs_list = [_probs_to_freqs(probs[i]) for i in range(seq_len)]

        payload = _arithmetic_encode(symbols, freqs_list)
        # tiny header so decompress() knows the token-grid shape (h, w each < 256 here)
        header = bytes([h, w])
        return header + payload

    def decompress(self, x: bytes) -> torch.Tensor:
        """
        Decompress a tensor into a PIL image.
        You may assume the output image is 150 x 100 pixels.
        """
        device = next(self.autoregressive.parameters()).device
        h, w = x[0], x[1]
        payload = x[2:]
        seq_len = h * w

        def get_freqs(decoded_so_far: list[int]) -> list[int]:
            cur = torch.zeros((1, seq_len), dtype=torch.long, device=device)
            if decoded_so_far:
                cur[0, : len(decoded_so_far)] = torch.tensor(decoded_so_far, dtype=torch.long, device=device)
            with torch.no_grad():
                logits, _ = self.autoregressive(cur.view(1, h, w))
            i = len(decoded_so_far)
            n_tokens = logits.shape[-1]
            probs = torch.softmax(logits.view(seq_len, n_tokens)[i].double(), dim=-1)
            return _probs_to_freqs(probs)

        decoded = _arithmetic_decode(payload, seq_len, get_freqs)
        tokens = torch.tensor(decoded, dtype=torch.long, device=device).view(1, h, w)
        with torch.no_grad():
            img = self.tokenizer.decode_index(tokens)  # (1, H, W, C)
        return img[0]


def compress(tokenizer: Path, autoregressive: Path, image: Path, compressed_image: Path):
    """
    Compress images using a pre-trained model.

    tokenizer: Path to the tokenizer model.
    autoregressive: Path to the autoregressive model.
    images: Path to the image to compress.
    compressed_image: Path to save the compressed image tensor.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tk_model = cast(Tokenizer, torch.load(tokenizer, weights_only=False).to(device))
    ar_model = cast(Autoregressive, torch.load(autoregressive, weights_only=False).to(device))
    cmp = Compressor(tk_model, ar_model)

    x = torch.tensor(np.array(Image.open(image)), dtype=torch.uint8, device=device)
    cmp_img = cmp.compress(x.float() / 255.0 - 0.5)
    with open(compressed_image, "wb") as f:
        f.write(cmp_img)


def decompress(tokenizer: Path, autoregressive: Path, compressed_image: Path, image: Path):
    """
    Decompress images using a pre-trained model.

    tokenizer: Path to the tokenizer model.
    autoregressive: Path to the autoregressive model.
    compressed_image: Path to the compressed image tensor.
    images: Path to save the image to compress.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tk_model = cast(Tokenizer, torch.load(tokenizer, weights_only=False).to(device))
    ar_model = cast(Autoregressive, torch.load(autoregressive, weights_only=False).to(device))
    cmp = Compressor(tk_model, ar_model)

    with open(compressed_image, "rb") as f:
        cmp_img = f.read()

    x = cmp.decompress(cmp_img)
    img = Image.fromarray(((x + 0.5) * 255.0).clamp(min=0, max=255).byte().cpu().numpy())
    img.save(image)


if __name__ == "__main__":
    from fire import Fire

    Fire({"compress": compress, "decompress": decompress})