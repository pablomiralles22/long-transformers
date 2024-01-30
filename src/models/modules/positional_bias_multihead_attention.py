import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.attention_head_handler import AttentionHeadHandler
from src.utils.rotary_embedding import RotaryEmbedding

class PositionalBiasMultiheadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        max_len: int,
        shared_bias: bool = False,
        vdim=None,
        kdim=None,
    ):
        assert d_model % nhead == 0, "Error: the embedding dimension should be divisible by the number of heads"

        super().__init__()
        
        self.d_model = d_model
        self.nhead = nhead
        
        vdim = d_model if vdim is None else vdim
        kdim = d_model if kdim is None else kdim

        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(kdim, d_model, bias=False)
        self.W_V = nn.Linear(vdim, d_model, bias=True)

        self.W_O = nn.Linear(d_model, d_model, bias=True)

        self.max_len = max_len
        if shared_bias:
            self.q_bias, self.k_bias = PositionalBiasMultiheadAttention.get_bias(nhead, max_len, d_model)
        else:
            self.q_bias = nn.Parameter(torch.empty(nhead, max_len, d_model // nhead))
            self.k_bias = nn.Parameter(torch.empty(nhead, max_len, d_model // nhead))
            self.reset_parameters()

    @classmethod
    def get_bias(cls, nhead, max_len, d_model):
        if hasattr(cls, 'cls_q_bias') is False:
            cls.cls_q_bias = nn.Parameter(torch.empty(nhead, max_len, d_model // nhead)) 
            nn.init.xavier_uniform_(cls.cls_q_bias)
        if hasattr(cls, 'cls_k_bias') is False:
            cls.cls_k_bias = nn.Parameter(torch.empty(nhead, max_len, d_model // nhead))
            nn.init.xavier_uniform_(cls.cls_k_bias)
        return cls.cls_q_bias, cls.cls_k_bias

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.q_bias)
        nn.init.xavier_uniform_(self.k_bias)
        
    def forward(
        self,
        queries,  # [B, L1, D]
        values=None,  # [B, L2, VDIM]
        keys=None,  # [B, L2, KDIM]
        key_attention_mask=None,  # [B, L2]
    ):  # -> [B, L1, D]
        values = queries if values is None else values
        keys = values if keys is None else keys

        B, L1, D = queries.shape
        _, L2, _ = keys.shape

        queries_proj = self.W_Q(queries)  # [B, L1, D]
        keys_proj = self.W_K(keys)  # [B, L2, D]
        values_proj = self.W_V(values)  # [B, L2, D]

        Q = AttentionHeadHandler.separate_heads(queries_proj, self.nhead)  # [B, H, L1, D/H]
        K = AttentionHeadHandler.separate_heads(keys_proj, self.nhead)  # [B, H, L2, D/H]
        V = AttentionHeadHandler.separate_heads(values_proj, self.nhead)  # [B, H, L2, D/H]

        # positional_bias = RotaryEmbedding.apply(self.qk_bias, freq=10000)  # [B, H, L, D/H]
        # positional_bias = self.qk_bias  # [B, H, L, D/H]

        Q = Q + self.q_bias[..., :L1, :]  # [B, H, L1, D/H]
        K = K + self.k_bias[..., :L2, :]  # [B, H, L2, D/H]

        # Q = RotaryEmbedding.apply(Q, freq=10000)  # [B, H, L1, D/H]
        # K = RotaryEmbedding.apply(K, freq=10000)

        if key_attention_mask is not None:
            key_attention_mask = key_attention_mask.unsqueeze(1).unsqueeze(2).bool()  # [B, 1, 1, L2]
        
        heads = F.scaled_dot_product_attention(Q, K, V, attn_mask=key_attention_mask)  # [B, H, L, D/H]

        return self.W_O(AttentionHeadHandler.join_heads(heads))
