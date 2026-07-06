import abc
import torch

def load() -> torch.nn.Module:
    from pathlib import Path
    model_name = "AutoregressiveModel"
    model_path = Path(__file__).parent / f"{model_name}.pth"
    print(f"Loading {model_name} from {model_path}")
    return torch.load(model_path, weights_only=False)


class Autoregressive(abc.ABC):
    """
    Base class for all autoregressive token models.
    """

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Take a tensor x (B, h, w) of integer tokens as input.
        Produce a probability distribution (as logits) over the next token
        for each position, i.e. logits[:, i] should depend only on
        x[:, :i] (strictly earlier tokens), never on x[:, i] itself.
        """

    @abc.abstractmethod
    def generate(self, B: int = 1, h: int = 30, w: int = 20, device=None) -> torch.Tensor:
        """
        Generate a new set of tokens of size (B, h, w)
        """


class AutoregressiveModel(torch.nn.Module, Autoregressive):
    """
    Implement a causal transformer that predicts the next token given all previous ones.

    IMPORTANT: this class embeds tokens and then shifts the embedded sequence right by one
    (prepending a learned BOS embedding) BEFORE running it through the causal transformer.
    Without this shift, the causal mask alone is not enough: position i would still see its
    own token's embedding as input while being asked to predict that very token, so the model
    can "cheat" via the residual stream instead of actually learning to predict from context.
    That bug shows up as a suspiciously fast drop in training loss that doesn't correspond to
    real generative quality (generate() produces garbage even though train/loss looks great).
    """

    def __init__(self, d_latent: int = 128, n_tokens: int = 2**10):
        super().__init__()
        self.d_latent = d_latent
        self.n_tokens = n_tokens
        self.embedding = torch.nn.Embedding(n_tokens, d_latent)
        # Learned "beginning of sequence" embedding used at position 0 (since there is no
        # previous token to shift in from).
        self.bos = torch.nn.Parameter(torch.zeros(1, 1, d_latent))
        self.transformer = torch.nn.TransformerEncoderLayer(
            d_model=d_latent, nhead=4, dim_feedforward=d_latent*2,
            dropout=0.0, activation='gelu', batch_first=True, norm_first=True
        )
        self.output = torch.nn.Linear(d_latent, n_tokens)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        B, h, w = x.shape
        seq_len = h * w
        x_flat = x.view(B, seq_len)
        emb = self.embedding(x_flat)  # (B, seq_len, d_latent)

        # Shift right by one and prepend BOS, so position i's *input* is the embedding of
        # token i-1 (or BOS for i=0). Combined with the causal mask, logits[:, i] then only
        # ever depend on tokens strictly before i.
        bos = self.bos.expand(B, 1, -1)
        emb_shifted = torch.cat([bos, emb[:, :-1]], dim=1)

        mask = torch.nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)
        transformed = self.transformer(emb_shifted, src_mask=mask)
        logits = self.output(transformed).view(B, h, w, self.n_tokens)
        return logits, {}

    def generate(self, B: int = 1, h: int = 20, w: int = 30, device=None) -> torch.Tensor:
        if device is None:
            device = next(self.parameters()).device
        seq_len = h * w
        tokens = torch.zeros((B, seq_len), dtype=torch.long, device=device)
        for i in range(seq_len):
            # forward()'s internal shift + causal mask guarantee logits[:, i] only depends on
            # tokens[:, :i], so the (not-yet-generated) placeholder zeros at positions >= i
            # have no effect on the prediction for position i.
            logits, _ = self.forward(tokens.view(B, h, w))
            next_logits = logits.view(B, seq_len, self.n_tokens)[:, i, :]
            probs = torch.nn.functional.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, 1).squeeze(1)
            tokens[:, i] = next_token
        return tokens.view(B, h, w)