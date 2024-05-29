import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.attention_head_handler import AttentionHeadHandler

class NymstromAttention(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        bias=True,
        vdim=None,
        kdim=None,
        qk_dim_out=None,
        v_dim_out=None,
        bottleneck_dim=128,
        eps=1e-4,
    ):
        assert d_model % nhead == 0, "Error: the embedding dimension should be divisible by the number of heads"

        super().__init__()
        
        self.d_model = d_model
        self.nhead = nhead
        self.eps = eps
        
        vdim = d_model if vdim is None else vdim
        kdim = d_model if kdim is None else kdim
        qk_dim_out = d_model if qk_dim_out is None else qk_dim_out * nhead
        v_dim_out = d_model if v_dim_out is None else v_dim_out * nhead

        self.bottleneck_vectors = nn.Parameter(torch.randn(nhead, bottleneck_dim, qk_dim_out // nhead))

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
        device = queries.device
        
        values = queries if values is None else values
        keys = values if keys is None else keys

        Q = self.W_Q(queries)  # [B, L1, D]
        K = self.W_K(keys)  # [B, L2, D]
        V = self.W_V(values)  # [B, L2, D]

        Q = AttentionHeadHandler.separate_heads(Q, self.nhead)  # [B, H, L1, Dqk/H]
        K = AttentionHeadHandler.separate_heads(K, self.nhead)  # [B, H, L2, Dqk/H]
        V = AttentionHeadHandler.separate_heads(V, self.nhead)  # [B, H, L2, Dv/H]


        nymstrom_mat_K = torch.exp(torch.einsum('hmd, bhld -> bhml', self.bottleneck_vectors, K))  # [B, H, N, L2]
        nymstrom_mat_K = torch.masked_fill(nymstrom_mat_K, key_attention_mask.unsqueeze(1).unsqueeze(1) == 0, 0.0)
        nymstrom_mat_Q = torch.exp(torch.einsum('hnd, bhld -> bhln', self.bottleneck_vectors, Q))  # [B, H, L1, N]
        nymstrom_mat = torch.exp(torch.einsum('hnd, hmd -> hnm', self.bottleneck_vectors, self.bottleneck_vectors))  # [H, N, N]

        eps_eye = torch.eye(nymstrom_mat.shape[-1]).to(device) * self.eps
        # nymstrom_mat_inv = torch.inverse(nymstrom_mat + eps_eye)  # [H, N, N]

        adjusted_Q = torch.linalg.solve(nymstrom_mat + eps_eye, nymstrom_mat_Q, left=False)  # [B, H, L1, N]

        # nmat_Q 2 nmat_inv @ nmat_K @ V
        # adjusted_Q = torch.einsum('bhln, hnm -> bhlm', nymstrom_mat_Q, nymstrom_mat_inv)  # [B, H, L1, N]
        KV = torch.einsum('bhml, bhle -> bhme', nymstrom_mat_K, V)  # [B, H, N, Dv/H]
        unnormalized_heads = torch.einsum('bhlm, bhme -> bhle', adjusted_Q, KV)  # [B, H, L1, Dv/H]

        normalization = torch.einsum('bhlm, bhm -> bhl', adjusted_Q, nymstrom_mat_K.sum(dim=-1)) + self.eps  # [B, H, L1]

        normalized_heads = unnormalized_heads / normalization.unsqueeze(-1)

        return self.W_O(AttentionHeadHandler.join_heads(normalized_heads))
