import abc
import torch

def load() -> torch.nn.Module:
    from pathlib import Path
    model_name = "AutoregressiveModel"
    model_path = Path(__file__).parent / f"{model_name}.pth"
    print(f"Loading {model_name} from {model_path}")
    return torch.load(model_path, weights_only=False)

class AutoregressiveModel(torch.nn.Module):
    def __init__(self, d_latent: int = 128, n_tokens: int = 2**10):
        super().__init__()
        self.d_latent = d_latent
        self.n_tokens = n_tokens
        self.embedding = torch.nn.Embedding(n_tokens, d_latent)
        self.transformer = torch.nn.TransformerEncoderLayer(
            d_model=d_latent, nhead=4, dim_feedforward=d_latent*2,
            dropout=0.0, activation='gelu', batch_first=True, norm_first=True
        )
        self.output = torch.nn.Linear(d_latent, n_tokens)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        B, h, w = x.shape
        seq_len = h * w
        x_flat = x.view(B, seq_len)
        emb = self.embedding(x_flat)
        mask = torch.nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)
        transformed = self.transformer(emb, src_mask=mask)
        logits = self.output(transformed).view(B, h, w, self.n_tokens)
        return logits, {}

    def generate(self, B: int = 1, h: int = 20, w: int = 30, device=None) -> torch.Tensor:
        if device is None:
            device = next(self.parameters()).device
        seq_len = h * w
        tokens = torch.zeros((B, seq_len), dtype=torch.long, device=device)
        for i in range(seq_len):
            curr = tokens[:, :i+1]
            # Reshape to (B, 1, current_len) for forward
            if i == 0:
                curr_reshaped = torch.zeros(B, 1, 1, dtype=torch.long, device=device)
            else:
                curr_reshaped = curr.view(B, 1, i+1)
            logits, _ = self.forward(curr_reshaped)
            # Last position
            next_logits = logits[:, 0, -1, :]
            probs = torch.nn.functional.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, 1).squeeze(1)
            tokens[:, i] = next_token
        return tokens.view(B, h, w)