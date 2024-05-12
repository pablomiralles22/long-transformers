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
        dtype = queries.dtype

        # key_attention_mask = key_attention_mask.to(dtype) if key_attention_mask is not None else torch.ones(B, L2, dtype=dtype)

        values = queries if values is None else values
        keys = values if keys is None else keys

        queries_proj = self.W_Q(queries)  # [B, L1, D]
        keys_proj = self.W_K(keys)  # [B, L2, D]
        values_proj = self.W_V(values)  # [B, L2, D]

        Q = AttentionHeadHandler.separate_heads(queries_proj, self.nhead)  # [B, H, L1, D/H]
        K = AttentionHeadHandler.separate_heads(keys_proj, self.nhead)  # [B, H, L2, D/H]
        V = AttentionHeadHandler.separate_heads(values_proj, self.nhead)  # [B, H, L2, D/H]

        Q = F.sigmoid(Q)  # [B, H, L1, D/H]
        K = F.sigmoid(K)  # [B, H, L2, D/H]

        # Q, K = AmplitudeEmbedding.apply(Q, K)

        if key_attention_mask is not None:
            K = K.masked_fill(~key_attention_mask.view(B, 1, L2, 1).bool(), 0.)
            V = V.masked_fill(~key_attention_mask.view(B, 1, L2, 1).bool(), 0.)

        heads_normalizer = torch.einsum("bhld,bhd->bhl", Q, K.sum(dim=-2))  # [B, H, L1]

        scaled_Q = Q / heads_normalizer.unsqueeze(-1)  # [B, H, L1, D/H]

        key_values = torch.einsum("bhld,bhlc->bhdc", K, V)  # [B, H, D/H, D/H]
        heads = torch.einsum("bhld,bhdc->bhlc", scaled_Q, key_values)  # [B, H, L1, D/H]

        return self.W_O(AttentionHeadHandler.join_heads(heads))
