import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.masked import masked_tensor
from src.utils.alibi_biases import ALiBiBiases
from src.utils.attention_head_handler import AttentionHeadHandler

class ALiBiMultiheadAttention(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        bias=True,
        vdim=None,
        kdim=None,
        max_len=1025,
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

        self.positional_biases = nn.Parameter(torch.empty((nhead, 2 * max_len - 1)))
        self.reset_parameters()

    def reset_parameters(self):
        L = (self.positional_biases.shape[1] + 1) // 2
        data = - torch.abs(L - 1 - torch.arange(0, 2 * L - 1)) / L
        self.positional_biases = nn.Parameter(data.repeat(self.nhead, 1))

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
        H = self.nhead

        queries_proj = self.W_Q(queries)  # [B, L1, D]
        keys_proj = self.W_K(keys)  # [B, L2, D]
        values_proj = self.W_V(values)  # [B, L2, D]

        Q = AttentionHeadHandler.separate_heads(queries_proj, self.nhead)  # [B, H, L1, D/H]
        K = AttentionHeadHandler.separate_heads(keys_proj, self.nhead)  # [B, H, L2, D/H]
        V = AttentionHeadHandler.separate_heads(values_proj, self.nhead)  # [B, H, L2, D/H]

        # attn_mask = ALiBiBiases.apply(L1, self.nhead, L2, device=queries.device, dtype=queries.dtype)  # [H, L1, L2]
        attn_mask = (
            self.positional_biases
                .unfold(1, L2, 1)
                .view(H, L1, L2)
                .flip(dims=(-1,))
        )
        # if key_attention_mask is not None:
        #     # key_attention_mask = key_attention_mask.unsqueeze(1).unsqueeze(2).bool()  # [B, 1, 1, L2]
        #     # attn_mask = attn_mask.masked_fill(~key_attention_mask, float("-inf"))
        #     key_attention_mask = (
        #         key_attention_mask
        #             .bool()
        #             .unsqueeze(1)
        #             .unsqueeze(2)
        #             .expand(B, H, L1, L2)
        #     )
        #     attn_mask = masked_tensor(attn_mask, ~key_attention_mask)
        
        heads = F.scaled_dot_product_attention(Q, K, V, attn_mask=attn_mask)  # [B, H, L, D/H]
        
        # scaled dot product self attention
        # attn_mask = self.positional_biases.unfold(1, L2, 1).view(H, L1, L2)
        # attn_mat = torch.matmul(Q, K.transpose(-1, -2)) / (D ** 0.5)  # [B, H, L1, L2]
        # attn_mat = attn_mat + attn_mask  # [B, H, L1, L2]
        # if key_attention_mask is not None:
        #     key_attention_mask = key_attention_mask.unsqueeze(1).unsqueeze(2).bool()  # [B, 1, 1, L2]
        #     attn_mat.masked_fill_(~key_attention_mask, float("-inf"))
        # print(attn_mat)
        # attn_weights = F.softmax(attn_mat, dim=-1)
        # heads = torch.matmul(attn_weights, V)

        return self.W_O(AttentionHeadHandler.join_heads(heads))
