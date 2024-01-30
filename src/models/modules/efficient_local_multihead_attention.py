import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.attention_head_handler import AttentionHeadHandler

class EfficientLocalMultiheadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        local_attention_window: int,
        bias: bool = True,
        vdim=None,
        kdim=None,
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

        self.kernel_logits = nn.Parameter(torch.Empty(nhead, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1))
        self.alpha_logits = nn.Parameter(torch.Empty(nhead, 1, 1))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.kernel_logits)
        nn.init.xavier_uniform_(self.alpha_logits)

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
        

        if key_attention_mask is not None:
            key_attention_mask = key_attention_mask.unsqueeze(1).unsqueeze(2).bool()  # [B, 1, 1, L2]
        
        global_heads = F.scaled_dot_product_attention(Q, K, V, attn_mask=key_attention_mask)  # [B, H, L, D/H]
        local_heads = self.__locality_based_attention(V, key_attention_mask)  # [B, H, L, D/H]

        alpha = torch.sigmoid(self.alpha_logits)  # [H, 1, 1]
        heads = alpha * global_heads + (1 - alpha) * local_heads

        return self.W_O(AttentionHeadHandler.join_heads(heads))

    def __locality_based_attention(
        self,
        V,  # [B, H, L2, D/H]
        key_attention_mask=None,  # [B, 1, 1, L2]
    ):
        B, H, L2, D = V.shape

        V = V * key_attention_mask.view(B, 1, L2, 1)

        kernel = torch.sigmoid(self.kernel_logits)
