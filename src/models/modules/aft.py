import torch.nn as nn
import torch.nn.functional as F

from typing import Optional
from opt_einsum import contract

class AFT(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,  #  useless
        vdim=None,
        kdim=None,
        internal_dim_multiplier: int = 2,
        # pos_bias_window: Optional[int] = None,
    ):
        # assert d_model % nhead == 0, "Error: the embedding dimension should be divisible by the number of heads"

        super().__init__()
        
        self.d_model = d_model
        # self.nhead = nhead
        
        vdim = d_model if vdim is None else vdim
        kdim = d_model if kdim is None else kdim

        self.internal_dim = d_model * internal_dim_multiplier

        self.W_Q = nn.Linear(d_model, self.internal_dim)
        self.W_K = nn.Linear(kdim, self.internal_dim)
        self.W_V = nn.Linear(vdim, self.internal_dim)

        self.W_O = nn.Linear(self.internal_dim, d_model)

    #     if pos_bias_window is not None:
    #         param_sz = 2 * pos_bias_window + 1
    #         self.pos_bais = nn.Parameter(torch.empty(param_sz, d_model))

    # def reset_parameters(self):
    #     if hasattr(self, "pos_bais"):
    #         nn.init.uniform_(self.pos_bais, a=-0.1, b=0.1)

    def forward(
        self,
        queries,  # [B, L1, D]
        values=None,  # [B, L2, VDIM]
        keys=None,  # [B, L2, KDIM]
        key_attention_mask=None,  # [B, L2]
    ):  # -> [B, L1, D]
        values = queries if values is None else values
        keys = values if keys is None else keys

        _, _, QKDIM = keys.shape

        Q = F.sigmoid(self.W_Q(queries)) / QKDIM  # [B, L1, D]
        K = F.softmax(self.W_K(keys), dim=1)  # [B, L2, D]
        V = self.W_V(values)  # [B, L2, D]
        
        # if key_attention_mask is not None:
        #     key_attention_mask = key_attention_mask.unsqueeze(1).unsqueeze(2).bool()  # [B, 1, 1, L]

        # heads = F.scaled_dot_product_attention(Q, K, V, attn_mask=key_attention_mask)  # [B, H, L, D/H]
        proj = contract("bld, brd, brd, br -> bld", Q, K, V, key_attention_mask.float())

        return self.W_O(proj)
