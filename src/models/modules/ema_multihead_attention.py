import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Literal
from src.models.modules.ema import EMA
from src.utils.rotary_embedding import RotaryEmbedding

class EMAMultiheadAttention(nn.Module):
    """
    Warning: for now it only makes sense with queries == values == keys
    """
    def __init__(
        self,
        d_model,
        nhead,  # ignore, just for compatibility
        bias=True,
        vdim=None,
        kdim=None,
        qk_dim_out=None,
        v_dim_out=None,
        dropout=0.0,
        ema_dim: int = 1,
        ema_kernel_size: int = 15,
        direction: Literal["forward", "backward", "bidirectional"] = "forward",
    ):
        super().__init__()
        
        self.d_model = d_model
        
        vdim = d_model if vdim is None else vdim
        kdim = d_model if kdim is None else kdim
        qk_dim_out = d_model if qk_dim_out is None else qk_dim_out * nhead
        v_dim_out = d_model if v_dim_out is None else v_dim_out * nhead

        self.dropout_p = dropout

        self.ema = EMA(d_model, ema_dim, ema_kernel_size, direction=direction)
        self.omega = nn.Parameter(torch.Tensor(d_model))
        self.dropout_ema = nn.Dropout(dropout)
        # self.x_prime_norm = nn.LayerNorm(d_model)

        self.W_Z = nn.Linear(d_model, qk_dim_out, bias=bias)
        self.dropout_z = nn.Dropout(dropout)

        self.kappa_Q = nn.Parameter(torch.Tensor(qk_dim_out))
        self.mu_Q = nn.Parameter(torch.Tensor(qk_dim_out))
        self.kappa_K = nn.Parameter(torch.Tensor(qk_dim_out))
        self.mu_K = nn.Parameter(torch.Tensor(qk_dim_out))

        self.W_V = nn.Linear(vdim, v_dim_out, bias=bias)

        # self.W_O = nn.Linear(v_dim_out, d_model, bias=bias)

        self.W_gamma = nn.Linear(d_model, v_dim_out, bias=bias)
        self.W_phi = nn.Linear(d_model, d_model, bias=bias)

        self.W_h = nn.Linear(d_model, d_model, bias=bias)
        self.U_h = nn.Linear(v_dim_out, d_model, bias=False)
        self.dropout_h = nn.Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self):
        std = 0.02
        nn.init.normal_(self.omega, mean=0.0, std=std)
        nn.init.normal_(self.kappa_Q, mean=0.0, std=std)
        nn.init.constant_(self.mu_Q, 0.0)
        nn.init.normal_(self.kappa_K, mean=0.0, std=std)
        nn.init.constant_(self.mu_K, 0.0)

    def forward(
        self,
        queries,  # [B, L1, D]
        values=None,  # [B, L2, VDIM]
        keys=None,  # [B, L2, KDIM]
        key_attention_mask=None,  # [B, L]
    ):  # -> [B, L, D]

        assert (values is None or values is queries) and (keys is None or keys is queries), "EMA only makes sense with queries == values == keys"
        
        embeddings = queries

        x = embeddings.masked_fill(~key_attention_mask.unsqueeze(-1).bool(), 0)  # [B, L, D] set padding to 0
        x_prime = F.silu(
            # self.x_prime_norm(
            self.dropout_ema(self.ema(x)) + x * self.omega
            # )
        )  # [B, L, D]
        z = self.dropout_z(F.silu(self.W_Z(x_prime)))  # [B, L, Z]

        queries_proj = self.kappa_Q * z + self.mu_Q  # [B, L, D]
        keys_proj = self.kappa_K * z + self.mu_K  # [B, L, D]
        values_proj = self.W_V(x)  # [B, L, D]
    
        # Apply rotary, as it is cheaper than relative biases in the attention scores
        Q = RotaryEmbedding.apply(queries_proj)  # [B, L, D]
        K = RotaryEmbedding.apply(keys_proj)  # [B, L, D]
        V = F.silu(values_proj)
        
        if key_attention_mask is not None:
            key_attention_mask = key_attention_mask.unsqueeze(1).bool()  # [B, 1, L]
        
        O = F.scaled_dot_product_attention(Q, K, V, attn_mask=key_attention_mask)  # [B, L, D]

        # output = self.W_O(O)

        gamma = F.silu(self.W_gamma(x_prime))  # [B, L, D]
        hat = self.dropout_h(F.silu(self.W_h(x_prime) + self.U_h(gamma * O)))  # [B, L, D]

        phi = F.sigmoid(self.W_phi(x_prime))

        return phi * hat + (1 - phi) * x
