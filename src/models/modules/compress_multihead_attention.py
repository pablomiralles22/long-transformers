import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.attention_head_handler import AttentionHeadHandler

class CompressMultiheadAttention(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        bias=True,
        vdim=None,
        kdim=None,
        bottleneck_dim: int = -1,  # I refer to this dimension as C
    ):
        assert d_model % nhead == 0, "Error: the embedding dimension should be divisible by the number of heads"

        super().__init__()
        
        self.d_model = d_model
        self.nhead = nhead
        self.bottleneck_dim = bottleneck_dim if bottleneck_dim > 0 else d_model
        
        vdim = d_model if vdim is None else vdim
        kdim = d_model if kdim is None else kdim

        self.W_Q = nn.Linear(d_model, d_model, bias=bias)
        self.W_K = nn.Linear(kdim, d_model, bias=bias)
        self.W_V = nn.Linear(vdim, d_model, bias=bias)

        self.W_O = nn.Linear(d_model, d_model, bias=bias)

        self.c = nn.Parameter(torch.empty(1, nhead, self.bottleneck_dim, (d_model // nhead)))  # [H, C, D/H]
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.c)

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

        Q = AttentionHeadHandler.separate_heads(queries_proj, self.nhead)  # [B, H, L1, D/H]
        K = AttentionHeadHandler.separate_heads(keys_proj, self.nhead)  # [B, H, L2, D/H]
        V = AttentionHeadHandler.separate_heads(values_proj, self.nhead)  # [B, H, L2, D/H]
        
        if key_attention_mask is not None:
            key_attention_mask = key_attention_mask.unsqueeze(1).unsqueeze(2).bool()  # [B, 1, 1, L2]
        
        step1 = F.scaled_dot_product_attention(self.c, K, V, attn_mask=key_attention_mask)  # [B, H, C, D/H]
        heads = F.scaled_dot_product_attention(Q, self.c, step1)  # [B, H, L1, D/H]

        # ################ step 1 ################
        # attn_scores_1 = torch.einsum("bhld,bhcd->bhlc", Q, self.c) / (self.d_model ** 0.5)
        # attn_mat_1 = F.softmax(attn_scores_1, dim=-1)  # [B, H, L1, C]

        # ################ step 2 ################
        # attn_scores_2 = torch.einsum("bhcd,bhld->bhcl", self.c, K) / (self.d_model ** 0.5)  # [B, H, C, L2]
        # if key_attention_mask is not None:
        #     attn_scores_2.masked_fill_(~key_attention_mask, float("-inf"))
        # attn_mat_2 = F.softmax(attn_scores_2, dim=-1)  # [B, H, C, L2]

        # # join
        # heads = attn_mat_1**0.5 @ (attn_mat_2**0.5 @ V)


        return self.W_O(AttentionHeadHandler.join_heads(heads))
