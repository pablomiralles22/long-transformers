import torch.nn as nn
import torch.nn.functional as F

from typing import Literal
from src.models.modules.ema import EMA
from src.utils.rotary_embedding import RotaryEmbedding
from src.utils.attention_head_handler import AttentionHeadHandler

class EMAMultiheadAttention(nn.Module):
    """
    Warning: for now it only makes sense with queries == values == keys
    """
    def __init__(
        self,
        d_model,
        nhead,
        bias=True,
        vdim=None,
        kdim=None,
        dropout=0.0,
        ema_dim: int = 1,
        ema_kernel_size: int = 15,
        direction: Literal["forward", "backward", "bidirectional"] = "forward",
    ):
        assert d_model % nhead == 0, "Error: the embedding dimension should be divisible by the number of heads"

        super().__init__()
        
        self.d_model = d_model
        self.nhead = nhead
        
        vdim = d_model if vdim is None else vdim
        kdim = d_model if kdim is None else kdim

        self.dropout_p = dropout

        self.ema = EMA(d_model, ema_dim, ema_kernel_size, direction=direction)
        self.dropout_ema = nn.Dropout(dropout)

        self.W_Z = nn.Linear(d_model, d_model, bias=bias)
        self.dropout_z = nn.Dropout(dropout)

        self.W_Q = nn.Linear(d_model, d_model, bias=bias)
        self.W_K = nn.Linear(kdim, d_model, bias=bias)
        self.W_V = nn.Linear(vdim, d_model, bias=bias)

        self.W_O = nn.Linear(d_model, d_model, bias=bias)

        self.W_gamma = nn.Linear(d_model, d_model, bias=bias)
        self.W_phi = nn.Linear(d_model, d_model, bias=bias)

        self.W_h = nn.Linear(d_model, d_model, bias=bias)
        self.U_h = nn.Linear(d_model, d_model, bias=False)
        self.dropout_h = nn.Dropout(dropout)


    def forward(
        self,
        embeddings,  # [B, L, D]
        key_attention_mask=None,  # [B, L]
    ):  # -> [B, L, D]

        x = embeddings * key_attention_mask.unsqueeze(-1)  # [B, L, D] set padding to 0
        x_prime = F.relu(self.dropout_ema(self.ema(x)) + x)  # [B, L, D]
        z = self.dropout_z(F.relu(self.W_Z(x_prime)))  # [B, L, D]

        queries_proj = self.W_Q(z)  # [B, L, D]
        keys_proj = self.W_K(z)  # [B, L, D]
        values_proj = self.W_V(x)  # [B, L, D]

        Q = RotaryEmbedding.apply(queries_proj)  # [B, L, D]
        K = RotaryEmbedding.apply(keys_proj)  # [B, L, D]
        V = F.relu(values_proj)

        # Q = AttentionHeadHandler.separate_heads(Q, self.nhead)  # [B, H, L, D/H]
        # K = AttentionHeadHandler.separate_heads(K, self.nhead)  # [B, H, L, D/H]
        # V = AttentionHeadHandler.separate_heads(values_proj, self.nhead)  # [B, H, L, D/H]
        
        if key_attention_mask is not None:
            # key_attention_mask = key_attention_mask.unsqueeze(1).unsqueeze(2).bool()  # [B, 1, 1, L]
            key_attention_mask = key_attention_mask.unsqueeze(1).bool()  # [B, 1, L]
        
        # heads = F.scaled_dot_product_attention(Q, K, V, attn_mask=key_attention_mask)  # [B, H, L, D/H]
        O = F.scaled_dot_product_attention(Q, K, V, attn_mask=key_attention_mask, dropout_p=self.dropout_p)  # [B, L, D]

        # output = self.W_O(AttentionHeadHandler.join_heads(heads))
        output = self.W_O(O)

        gamma = F.relu(self.W_gamma(x_prime))  # [B, L, D]
        hat = self.dropout_h(F.relu(self.W_h(x_prime) + self.U_h(gamma * output)))  # [B, L, D]

        phi = F.sigmoid(self.W_phi(x_prime))

        return phi * hat + (1 - phi) * x
