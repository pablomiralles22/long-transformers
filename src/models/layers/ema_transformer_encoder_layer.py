from typing import Optional, Literal
from torch import nn
from src.models.layers.layer import Layer
from src.models.modules.ema_multihead_attention import (
    EMAMultiheadAttention,
)


class EMATransformerEncoderLayer(Layer):
    def __init__(
        self,
        d_model,
        nhead=4,
        dropout=0.1,
        dim_feedforward=2048,
        activation_fn_cls=nn.ReLU,
        layer_norm_eps=1e-05,
        norm_first=True,
        ema_dim: Optional[int] = None,
        ema_kernel_size: int = 15,
        direction: Literal["forward", "backward", "bidirectional"] = "forward",
    ):
        super().__init__()

        self.mh_attention = EMAMultiheadAttention(
            d_model,
            nhead,
            ema_dim=ema_dim,
            ema_kernel_size=ema_kernel_size,
            dropout=dropout,
            direction=direction,
        )
        self.dropout_1 = nn.Dropout(dropout)
        self.layer_norm_1 = nn.LayerNorm(d_model, layer_norm_eps)

        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            activation_fn_cls(),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.layer_norm_2 = nn.LayerNorm(d_model, layer_norm_eps)

        self.norm_first = norm_first

    def forward(
        self,
        embeddings,  # (BATCH, LENGTH, d_model)
        attention_mask=None,  # (BATCH, LENGTH)
        token_type_ids=None,  # (BATCH, LENGTH)
    ):
        if self.norm_first is True:
            x = self.layer_norm_1(embeddings)
            x = self.mh_attention(x, key_attention_mask=attention_mask)
            x = embeddings + self.dropout_1(x)
            return x + self.ff(self.layer_norm_2(x))

        else:
            x = self.mh_attention(embeddings, key_attention_mask=attention_mask)
            x = self.dropout_1(x)
            x = self.layer_norm_1(embeddings + x)
            return self.layer_norm_2(x + self.ff(x))
