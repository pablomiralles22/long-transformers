import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.attention_head_handler import AttentionHeadHandler
from src.utils.rotary_embedding import RotaryEmbedding

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

        Q = self.W_Q(queries)  # [B, L1, D]
        K = self.W_K(keys)  # [B, L2, D]
        V = self.W_V(values)  # [B, L2, D]

        Q = AttentionHeadHandler.separate_heads(Q, self.nhead)  # [B, H, L1, Dqk/H]
        K = AttentionHeadHandler.separate_heads(K, self.nhead)  # [B, H, L2, Dqk/H]
        V = AttentionHeadHandler.separate_heads(V, self.nhead)  # [B, H, L2, Dv/H]

        log_B = torch.logsumexp(Q + torch.logsumexp(K, dim=-2, keepdim=True), dim=-1)  # [B, H, L1]

        log_V_plus = torch.log(torch.max(V, torch.zeros_like(V))).unsqueeze(-2)  # [B, H, L2, 1, Dv/H]
        log_V_minus = torch.log(-torch.min(V, torch.zeros_like(V))).unsqueeze(-2)  # [B, H, L2, 1, Dv/H]

        logsumexp_KV_plus = torch.logsumexp(K.unsqueeze(-1) + log_V_plus, dim=-3)  # [B, H, Dqk/H, Dv/H]
        logsumexp_KV_minus = torch.logsumexp(K.unsqueeze(-1) + log_V_minus, dim=-3)  # [B, H, Dqk/H, Dv/H]

        log_A_plus = torch.logsumexp(Q.unsqueeze(-1) + logsumexp_KV_plus.unsqueeze(-3), dim=-2)  # [B, H, L1, Dv/H]
        log_A_minus = torch.logsumexp(Q.unsqueeze(-1) + logsumexp_KV_minus.unsqueeze(-3), dim=-2)  # [B, H, L1, Dv/H]


        heads_plus = torch.exp(log_A_plus - log_B.unsqueeze(-1))  # [B, H, L1, Dv/H]
        heads_minus = torch.exp(log_A_minus - log_B.unsqueeze(-1))  # [B, H, L1, Dv/H]

        heads = heads_plus - heads_minus

        return self.W_O(AttentionHeadHandler.join_heads(heads))

        # Q = RotaryEmbedding.apply(Q)  # [B, L, D]
        # K = RotaryEmbedding.apply(K)  # [B, L, D]

        # Q = AttentionHeadHandler.separate_heads(Q, self.nhead)  # [B, H, L1, D/H]
        # K = AttentionHeadHandler.separate_heads(K, self.nhead)  # [B, H, L2, D/H]
        # V = AttentionHeadHandler.separate_heads(V, self.nhead)  # [B, H, L2, D/H]

        # Q = F.relu(Q) * 1e-3  # [B, H, L1, D/H]
        # K = F.relu(K) * 1e-3  # [B, H, L2, D/H]

        # # Q, K = AmplitudeEmbedding.apply(Q, K)

        # if key_attention_mask is not None:
        #     K = K.masked_fill(~key_attention_mask.view(B, 1, L2, 1).bool(), 0.)
        #     V = V.masked_fill(~key_attention_mask.view(B, 1, L2, 1).bool(), 0.)

        # heads_normalizer = torch.einsum("bhld,bhd->bhl", Q, K.sum(dim=-2))  # [B, H, L1]
        # scaled_Q = Q / heads_normalizer.unsqueeze(-1)  # [B, H, L1, D/H]

        # key_values = torch.matmul(K.transpose(-2, -1), V)  # [B, H, D/H, D/H]
        # heads = torch.matmul(scaled_Q, key_values)  # [B, H, L1, D/H]

        # return self.W_O(AttentionHeadHandler.join_heads(heads))
