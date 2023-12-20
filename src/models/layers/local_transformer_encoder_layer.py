from torch import nn
from src.models.layers.layer import Layer
from src.models.modules.local_multihead_self_attention import LocalMultiheadSelfAttention

class LocalTransformerEncoderLayer(Layer):
    def __init__(
        self,
        d_model, vdim=None, qkdim=None,
        nhead=4,
        dropout=0.1,
        dim_feedforward=2048,
        activation_fn_cls=nn.ReLU,
        layer_norm_eps=1e-05,
        freq=10000,
        window_size=512,
    ):
        super().__init__()
        
        self.mh_attention = LocalMultiheadSelfAttention(d_model, nhead, vdim=vdim, qkdim=qkdim, window_size=window_size, freq=freq)
        self.dropout_1 = nn.Dropout(dropout)
        self.layer_norm_1 = nn.LayerNorm(d_model, layer_norm_eps)
        
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            activation_fn_cls(),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.layer_norm_2 = nn.LayerNorm(d_model, layer_norm_eps)

    
    def forward(
        self,
        embeddings,  # (BATCH, LENGTH, d_model)
        attention_mask=None,  # (BATCH, LENGTH)
        token_type_ids=None,  # (BATCH, LENGTH)
    ):
        x = self.mh_attention(embeddings, attention_mask)
        x = self.dropout_1(x)
        x = self.layer_norm_1(embeddings + x)
        y = self.ff(x)
        return self.layer_norm_2(x + y)
        