import torch.nn as nn

class GLUReducer(nn.Module):
    def __init__(self, d_model: int, dropout_p: float = 0.1):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(d_model, 2*d_model),
            nn.LayerNorm(2*d_model),
            nn.Dropout(dropout_p),
            nn.GLU(),
        )

    def forward(self, x, _=None):
        # x: [B, L, D]
        # attention_mask: [B, L]
        return self.ff(x)