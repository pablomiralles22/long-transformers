import torch.nn as nn

from typing import Optional
from src.models.layers.layer import Layer
from src.models.modules.attention_module_builder import AttentionModuleBuilder

class TransformerEncoderLayer(Layer):
    def __init__(
        self,
        d_model,
        nhead=4,
        dropout=0.1,
        dim_feedforward=2048,
        activation_fn_cls=nn.SiLU,
        layer_norm_eps=1e-05,
        norm_first=True,
        attention_params: Optional[dict] = None,
    ):
        super().__init__()

        attention_params = {} if attention_params is None else attention_params
        self.mh_attention = AttentionModuleBuilder.build(d_model, nhead, attention_params)
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
            x = self.mh_attention(x, x, x, attention_mask)
            x = embeddings + self.dropout_1(x)
            return x + self.ff(self.layer_norm_2(x))

        else:
            x = self.mh_attention(embeddings, embeddings, embeddings, attention_mask)
            x = self.dropout_1(x)
            x = self.layer_norm_1(embeddings + x)
            return self.layer_norm_2(x + self.ff(x))
