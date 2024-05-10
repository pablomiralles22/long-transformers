import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.attention_head_handler import AttentionHeadHandler
from src.utils.amplitude_embedding import AmplitudeEmbedding

class PositiveMultiheadAttention(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        bias=True,
        vdim=None,
        kdim=None,
        qk_dim_out=None,
        v_dim_out=None,
    ):
        assert d_model % nhead == 0, "Error: the embedding dimension should be divisible by the number of heads"

        super().__init__()
        
        self.d_model = d_model
        self.nhead = nhead
        
        vdim = d_model if vdim is None else vdim
        kdim = d_model if kdim is None else kdim
        qk_dim_out = d_model if qk_dim_out is None else qk_dim_out * nhead
        v_dim_out = d_model if v_dim_out is None else v_dim_out * nhead

        self.W_Q = nn.Linear(d_model, qk_dim_out, bias=bias)
        self.W_K = nn.Linear(kdim, qk_dim_out, bias=bias)
        self.W_V = nn.Linear(vdim, v_dim_out, bias=bias)

        self.W_O = nn.Linear(v_dim_out, d_model, bias=bias)



    def forward(
        self,
        queries,  # [B, L1, D]
        values=None,  # [B, L2, VDIM]
        keys=None,  # [B, L2, KDIM]
        key_attention_mask=None,  # [B, L2]
    ):  # -> [B, L1, D]
        B, L1, D = queries.shape
        _, L2, _ = keys.shape

        values = queries if values is None else values
        keys = values if keys is None else keys

        queries_proj = self.W_Q(queries)  # [B, L1, D]
        keys_proj = self.W_K(keys)  # [B, L2, D]
        values_proj = self.W_V(values)  # [B, L2, D]

        Q = AttentionHeadHandler.separate_heads(queries_proj, self.nhead)  # [B, H, L1, D/H]
        K = AttentionHeadHandler.separate_heads(keys_proj, self.nhead)  # [B, H, L2, D/H]
        V = AttentionHeadHandler.separate_heads(values_proj, self.nhead)  # [B, H, L2, D/H]

        V_minus, V_plus = torch.min(V, torch.zeros_like(V)), torch.max(V, torch.zeros_like(V))

        Q = torch.sigmoid(Q)  # [B, H, L1, D/H]
        scaled_K = torch.sigmoid(K) / L2  # [B, H, L2, D/H]

        Q, scaled_K = AmplitudeEmbedding.apply(Q, scaled_K)

        key_values_plus = torch.einsum("bhld,bhlc,bl->bhdc", scaled_K, V_plus, key_attention_mask)  # [B, H, D/H, D/H]
        heads_plus = torch.einsum("bhld,bhdc->bhlc", Q, key_values_plus)  # [B, H, L1, D/H]
        heads_plus = F.normalize(heads_plus, p=2, dim=-1)

        key_values_minus = torch.einsum("bhld,bhlc,bl->bhdc", scaled_K, V_minus, key_attention_mask)  # [B, H, D/H, D/H]
        heads_minus = torch.einsum("bhld,bhdc->bhlc", Q, key_values_minus)  # [B, H, L1, D/H]
        heads_minus = F.normalize(heads_minus, p=2, dim=-1)

        return self.W_O(AttentionHeadHandler.join_heads(heads_plus + heads_minus))
