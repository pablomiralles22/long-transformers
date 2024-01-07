import torch.nn as nn
import torch.nn.functional as F

from src.utils.rotary_embedding import RotaryEmbedding
from src.utils.attention_head_handler import AttentionHeadHandler

class RotaryMultiheadAttention(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        bias=True,
        vdim=None,
        kdim=None,
        freq=10000,
    ):
        assert d_model % nhead == 0, "Error: the embedding dimension should be divisible by the number of heads"

        super().__init__()
        
        self.d_model = d_model
        self.nhead = nhead
        
        vdim = d_model if vdim is None else vdim
        kdim = d_model if kdim is None else kdim

        self.W_Q = nn.Linear(d_model, d_model, bias=bias)
        self.W_K = nn.Linear(kdim, d_model, bias=bias)
        self.W_V = nn.Linear(vdim, d_model, bias=bias)

        self.W_O = nn.Linear(d_model, d_model, bias=bias)

        self.freq = freq

    def forward(
        self,
        queries,  # [B, L1, D]
        values=None,  # [B, L2, VDIM]
        keys=None,  # [B, L2, KDIM]
        key_attention_mask=None,  # [B, L2]
    ):  # -> [B, L1, D]

        values = queries if values is None else values
        keys = values if keys is None else keys

        queries_proj = self.W_Q(queries)  # [B, L1, D]
        keys_proj = self.W_K(keys)  # [B, L2, D]
        values_proj = self.W_V(values)  # [B, L2, D]

        Q = RotaryEmbedding.apply(queries_proj, freq=self.freq)  # [B, H, L, D/H]
        K = RotaryEmbedding.apply(keys_proj, freq=self.freq)  # [B, H, L, D/H]

        Q = AttentionHeadHandler.separate_heads(Q, self.nhead)  # [B, H, L, D/H]
        K = AttentionHeadHandler.separate_heads(K, self.nhead)  # [B, H, L, D/H]
        V = AttentionHeadHandler.separate_heads(values_proj, self.nhead)  # [B, H, L, D/H]
        
        if key_attention_mask is not None:
            key_attention_mask = key_attention_mask.unsqueeze(1).unsqueeze(2).bool()  # [B, 1, 1, L]
        
        heads = F.scaled_dot_product_attention(Q, K, V, attn_mask=key_attention_mask)  # [B, H, L, D/H]

        return self.W_O(AttentionHeadHandler.join_heads(heads))
