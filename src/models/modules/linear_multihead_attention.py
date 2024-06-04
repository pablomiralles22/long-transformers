import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Literal
from src.utils.attention_head_handler import AttentionHeadHandler
# from src.models.functional.absdiff_attention_triton import AbsdiffAttention

EPS = 1e-5

class LinearMultiheadAttention(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        bias=True,
        vdim=None,
        kdim=None,
        qk_dim_out=None,
        v_dim_out=None,
        impl: Literal["std", "causal", "rel_pos", "absdiff"] = "std",
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

        if impl == "std":
            self.attn = std_linear_attention
        elif impl == "causal":
            self.attn = causal_linear_attention
        elif impl == "rel_pos":
            self.attn = RelPosAttention(nhead)
        # elif impl == "absdiff":
        #     self.attn = AbsdiffAttention()
        #     # make W_Q the identity function, as we won't use its output
        #     self.W_Q = nn.Identity()
        else:
            raise ValueError(f"Error: unknown implementation {impl}")

    def forward(
        self,
        queries,  # [B, L1, D]
        values=None,  # [B, L2, VDIM]
        keys=None,  # [B, L2, KDIM]
        key_attention_mask=None,  # [B, L2]
    ):  # -> [B, L1, D]
        values = queries if values is None else values
        keys = values if keys is None else keys

        Q = self.W_Q(queries)  # [B, L1, D]
        K = self.W_K(keys)  # [B, L2, D]
        V = self.W_V(values)  # [B, L2, D]

        Q = AttentionHeadHandler.separate_heads(Q, self.nhead)  # [B, H, L1, Dqk/H]
        K = AttentionHeadHandler.separate_heads(K, self.nhead)  # [B, H, L2, Dqk/H]
        V = AttentionHeadHandler.separate_heads(V, self.nhead)  # [B, H, L2, Dv/H]

        heads = self.attn(Q, K, V, key_attention_mask)
        return self.W_O(AttentionHeadHandler.join_heads(heads))


def std_linear_attention(
    Q, # [B, H, L1, Dqk/H]
    K, # [B, H, L2, Dqk/H]
    V, # [B, H, L2, Dv/H]
    key_attention_mask=None,  # [B, L2]
):
    B, _, L2, _ = K.shape

    original_dtype = Q.dtype
    K = K.to(torch.float32)
    V = V.to(torch.float32)
    Q = Q.to(torch.float32)

    feature_map_Q = torch.sigmoid
    feature_map_K = torch.sigmoid
    # feature_map_Q = F.softplus
    # feature_map_K = F.softplus

    Q = feature_map_Q(Q)  # [B, H, L1, Dqk/H]
    K = feature_map_K(K)  # [B, H, L2, Dqk/H]

    if key_attention_mask is not None:
        K = K.masked_fill(~key_attention_mask.view(B, 1, L2, 1).bool(), 0.)
        V = V.masked_fill(~key_attention_mask.view(B, 1, L2, 1).bool(), 0.)

    normalizer = 1. / (torch.einsum("bhld,bhd->bhl", Q, K.sum(dim=-2)) + EPS)  # [B, H, L1]
    KV = torch.einsum("bhld,bhle->bhde", K, V)  # [B, H, Dqk/H, Dv/H]
    O = torch.einsum("bhld,bhl,bhde->bhle", Q, normalizer, KV)
    return O.to(original_dtype)

def causal_linear_attention(
    Q, # [B, H, L, Dqk/H]
    K, # [B, H, L, Dqk/H]
    V, # [B, H, L, Dv/H]
    key_attention_mask=None,  # [B, L]
):
    B, _, L2, _ = K.shape
    _, _, L1, _ = Q.shape
    assert L1 == L2, "Error: the sequence length of the queries and keys should be the same for causal attention"

    Q = torch.exp(Q)  # [B, H, L, Dqk/H]
    K = torch.exp(K)  # [B, H, L, Dqk/H]

    if key_attention_mask is not None:
        K = K.masked_fill(~key_attention_mask.view(B, 1, L2, 1).bool(), 0.)
        V = V.masked_fill(~key_attention_mask.view(B, 1, L2, 1).bool(), 0.)
    
    cum_K = torch.cumsum(K, dim=-2)  # [B, H, L, Dqk/H]

    heads_normalizer = torch.einsum("bhld,bhld->bhl", Q, cum_K) + EPS  # [B, H, L1]
    scaled_Q = Q / heads_normalizer.unsqueeze(-1)  # [B, H, L, Dqk/H]

    cum_KV = torch.einsum("bhld,bhle->bhlde", K, V)  # [B, H, L, Dqk, Dv] TODO correct
    O = torch.einsum("bhld,bhlde->bhle", scaled_Q, cum_KV)  # [B, H, L, Dv/H]

    return O

class RelPosAttention(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.base_amp = nn.Parameter(torch.log((1. + torch.arange(0, num_heads)) / num_heads * 1e-2))

    def forward(
        self,
        Q, # [B, H, L1, Dqk/H]
        K, # [B, H, L2, Dqk/H]
        V, # [B, H, L2, Dv/H]
        key_attention_mask=None,  # [B, L2]
    ):
        B, H, L2, _ = K.shape
        _, _, L1, _ = Q.shape
        assert L1 == L2, "Error: the sequence length of the queries and keys should be the same for causal attention"
        device, dtype = Q.device, Q.dtype

        Q = torch.exp(Q)  # [B, H, L1, D/H]
        K = torch.exp(K)  # [B, H, L2, D/H]

        # base_amp = torch.arange(0, H, device=device, dtype=dtype).view(-1, 1) / H  # [H, 1]
        base_amp = torch.exp(self.base_amp).view(-1, 1)  # [H, 1]

        # forward masking
        Q_pos_forward = torch.exp(torch.arange(0, L1, device=device, dtype=dtype) * base_amp).view(1, H, L1)  # [1, H, L]
        K_pos_forward = torch.exp(-torch.arange(0, L2, device=device, dtype=dtype) * base_amp).view(1, H, L2)  # [1, H, L]

        # backward masking
        Q_pos_backward = torch.exp(-torch.arange(0, L1, device=device, dtype=dtype)  * base_amp).view(1, H, L1)  # [1, H, L]
        K_pos_backward = torch.exp(torch.arange(0, L2, device=device, dtype=dtype) * base_amp).view(1, H, L2)  # [1, H, L]

        if key_attention_mask is not None:
            K = K.masked_fill(~key_attention_mask.view(B, 1, L2, 1).bool(), 0.)  # [B, H, L, Dqk/H]
            K_pos_forward = K_pos_forward.masked_fill(~key_attention_mask.view(B, 1, L2).bool(), 0.)  # [B, H, L]
            K_pos_backward = K_pos_backward.masked_fill(~key_attention_mask.view(B, 1, L2).bool(), 0.)  # [B, H, L]
            V = V.masked_fill(~key_attention_mask.view(B, 1, L2, 1).bool(), 0.)  # [B, H, L, Dv/H]

        heads_normalizer = torch.einsum("bhld,bhd->bhl", Q, K.sum(dim=-2)) + EPS  # [B, H, L]
        KV = torch.matmul(K.transpose(-2, -1), V)  # [B, H, Dqk/H, Dv/H]

        # forward masking
        cum_K_pos_forward = torch.cumsum(K_pos_forward, dim=-1)  # [B, H, L]
        heads_normalizer += torch.einsum("bhl,bhl->bhl", Q_pos_forward, cum_K_pos_forward)  # [B, H, L]

        K_pos_forward_V = torch.einsum("bhl,bhle->bhle", K_pos_forward, V)  # [B, H, L, Dv/H]
        cum_K_pos_forward_V = torch.cumsum(K_pos_forward_V, dim=-2)  # [B, H, L, Dv/H]

        # backward masking
        cum_K_pos_backward = K_pos_backward.sum(dim=-1, keepdims=True) - torch.cumsum(K_pos_backward, dim=-1)  # [B, H, L]
        heads_normalizer += torch.einsum("bhl,bhl->bhl", Q_pos_backward, cum_K_pos_backward)  # [B, H, L]

        K_pos_backward_V = torch.einsum("bhl,bhle->bhle", K_pos_backward, V)  # [B, H, L, Dv/H]
        cum_K_pos_backward_V = K_pos_backward_V.sum(dim=-2, keepdims=True) - torch.cumsum(K_pos_backward_V, dim=-2)  # [B, H, L, Dv/H]

        # output
        scaled_Q = Q / heads_normalizer.unsqueeze(-1)  # [B, H, L, D/H]
        scaled_Q_pos_forward = Q_pos_forward / heads_normalizer  # [B, H, L]
        scaled_Q_pos_backward = Q_pos_backward / heads_normalizer  # [B, H, L]
        O = (
            torch.matmul(scaled_Q, KV) +
            torch.einsum("bhl,bhle->bhle", scaled_Q_pos_forward, cum_K_pos_forward_V) +
            torch.einsum("bhl,bhle->bhle", scaled_Q_pos_backward, cum_K_pos_backward_V)
        )  # [B, H, L, D/H]

        return O



    # def _impl_favor(
    #     self,
    #     Q,  # [B, H, L1, Dqk/H]
    #     K,  # [B, H, L2, Dqk/H]
    #     V,  # [B, H, L2, Dv/H]
    #     key_attention_mask=None,  # [B, L2]
    # ):
    #     B, _, L2, Dqk = K.shape
    #     device = K.device

    #     Q = RotaryEmbedding.apply(Q)
    #     K = RotaryEmbedding.apply(K)

    #     W_favor = torch.randn(self.nhead, self.bottleneck_dim, Dqk).to(device)  # [B, W, Dqk/H]

    #     Q_dot_W_favor = torch.einsum("bhld,hwd->bhlw", Q, W_favor)  # [B, H, L1, W]
    #     Q_feat = (
    #         torch.exp(-(Q**2).sum(dim=-1, keepdim=True)) *
    #         torch.cat([torch.exp(Q_dot_W_favor), torch.exp(-Q_dot_W_favor)], dim=-1) *
    #         ((self.bottleneck_dim * 2) ** -0.5)
    #     )
    #     K_dot_W_favor = torch.einsum("bhld,hwd->bhlw", K, W_favor)  # [B, H, L2, W]
    #     K_feat = (
    #         torch.exp(-(K**2).sum(dim=-1, keepdim=True)) *
    #         torch.cat([torch.exp(K_dot_W_favor), torch.exp(-K_dot_W_favor)], dim=-1) *
    #         ((self.bottleneck_dim * 2) ** -0.5)
    #     )

    #     if key_attention_mask is not None:
    #         K_feat = K_feat.masked_fill(~key_attention_mask.view(B, 1, L2, 1).bool(), 0.)
    #         V = V.masked_fill(~key_attention_mask.view(B, 1, L2, 1).bool(), 0.)

    #     heads_normalizer = torch.einsum("bhld,bhd->bhl", Q_feat, K_feat.sum(dim=-2)) + EPS  # [B, H, L1]
    #     scaled_Q_feat = Q_feat / heads_normalizer.unsqueeze(-1)  # [B, H, L1, D/H]

    #     key_values = torch.matmul(K_feat.transpose(-2, -1), V)  # [B, H, D/H, D/H]
    #     heads = torch.matmul(scaled_Q_feat, key_values)  # [B, H, L1, D/H]

    #     return self.W_O(AttentionHeadHandler.join_heads(heads))


    # def _impl2(
    #     self,
    #     Q, # [B, H, L1, Dqk/H]
    #     K, # [B, H, L2, Dqk/H]
    #     V, # [B, H, L2, Dv/H]
    #     key_attention_mask=None,  # [B, L2]
    # ):
    #     B, _, L2, _ = K.shape
        
    #     if key_attention_mask is not None:
    #         K = K.masked_fill(~key_attention_mask.view(B, 1, L2, 1).bool(), float("-inf"))
    #         V = V.masked_fill(~key_attention_mask.view(B, 1, L2, 1).bool(), 0.)

    #     Q = Q.softmax(dim=-1)  # [B, H, L1, D/H]
    #     K = K.softmax(dim=-2)  # [B, H, L2, D/H]

    #     key_values = torch.matmul(K.transpose(-2, -1), V)  # [B, H, D/H, D/H]
    #     heads = torch.matmul(Q, key_values)  # [B, H, L1, D/H]

    #     return self.W_O(AttentionHeadHandler.join_heads(heads))
    
    # def _impl3(
    #     self,
    #     Q, # [B, H, L1, Dqk/H]
    #     K, # [B, H, L2, Dqk/H]
    #     V, # [B, H, L2, Dv/H]
    #     key_attention_mask=None,  # [B, L2]
    # ):
    #     max_norm = 10.
    #     B, _, L2, _ = K.shape

    #     K_norm = K.norm(dim=-1, keepdim=True)  # [B, H, L2, 1]
    #     Q_norm = Q.norm(dim=-1, keepdim=True)  # [B, H, L1, 1]

    #     desired_K_norm = max_norm * (1. - torch.exp(-K_norm))  # [B, H, L2, 1]
    #     desired_Q_norm = max_norm * (1. - torch.exp(-Q_norm))  # [B, H, L1, 1]

    #     K = K * desired_K_norm / (K_norm + EPS)
    #     Q = Q * desired_Q_norm / (Q_norm + EPS)

    #     # append to Q and K a (max_norm**0.5) at the end
    #     K = torch.cat([K, torch.ones_like(K[..., :1]) * max_norm**0.5], dim=-1)  # [B, H, L2, D/H + 1]
    #     Q = torch.cat([Q, torch.ones_like(Q[..., :1]) * max_norm**0.5], dim=-1)  # [B, H, L1, D/H + 1]
        
    #     if key_attention_mask is not None:
    #         K = K.masked_fill(~key_attention_mask.view(B, 1, L2, 1).bool(), 0.)
    #         V = V.masked_fill(~key_attention_mask.view(B, 1, L2, 1).bool(), 0.)

    #     heads_normalizer = torch.einsum("bhld,bhd->bhl", Q, K.sum(dim=-2)) + EPS  # [B, H, L1]
    #     scaled_Q = Q / heads_normalizer.unsqueeze(-1)  # [B, H, L1, D/H]

    #     key_values = torch.matmul(K.transpose(-2, -1), V)  # [B, H, D/H, D/H]
    #     heads = torch.matmul(scaled_Q, key_values)  # [B, H, L1, D/H]

    #     return self.W_O(AttentionHeadHandler.join_heads(heads))


    # def _impl1_rotary(
    #     self,
    #     Q, # [B, H, L1, Dqk/H]
    #     K, # [B, H, L2, Dqk/H]
    #     V, # [B, H, L2, Dv/H]
    #     key_attention_mask=None,  # [B, L2]
    # ):
    #     B, _, L2, _ = K.shape

    #     Q = torch.exp(Q)  # [B, H, L1, D/H]
    #     K = torch.exp(K)  # [B, H, L2, D/H]

    #     Q_cos, Q_sen, K_cos, K_sen = self._apply_rotary_emb(Q, K)

    #     if key_attention_mask is not None:
    #         K_cos = K_cos.masked_fill(~key_attention_mask.view(B, 1, L2, 1).bool(), 0.)
    #         K_sen = K_sen.masked_fill(~key_attention_mask.view(B, 1, L2, 1).bool(), 0.)
    #         V = V.masked_fill(~key_attention_mask.view(B, 1, L2, 1).bool(), 0.)

    #     heads_normalizer = (
    #         torch.einsum("bhld,bhd->bhl", Q_cos, K_cos.sum(dim=-2)) +
    #         torch.einsum("bhld,bhd->bhl", Q_sen, K_sen.sum(dim=-2)) +
    #         EPS
    #     )  # [B, H, L1]
    #     scaled_Q_cos = Q_cos / heads_normalizer.unsqueeze(-1)  # [B, H, L1, D/H]
    #     scaled_Q_sen = Q_sen / heads_normalizer.unsqueeze(-1)  # [B, H, L1, D/H]

    #     key_values_cos = torch.matmul(K_cos.transpose(-2, -1), V)  # [B, H, D/H, D/H]
    #     heads_cos = torch.matmul(scaled_Q_cos, key_values_cos)  # [B, H, L1, D/H]

    #     key_values_sen = torch.matmul(K_sen.transpose(-2, -1), V)  # [B, H, D/H, D/H]
    #     heads_sen = torch.matmul(scaled_Q_sen, key_values_sen)

    #     heads = heads_cos + heads_sen

    #     return self.W_O(AttentionHeadHandler.join_heads(heads))
    
    # def _apply_rotary_emb(self, Q, K):
    #     _, _, L1, D = Q.shape
    #     _, _, L2, _ = K.shape
    #     device, dtype = Q.device, Q.dtype
    #     assert L1 == L2, "Error: the sequence length of the queries and keys should be the same for rotary embs"

    #     thetas = (
    #         torch.arange(0, D, dtype=dtype, device=device).view(1, -1) / D *
    #         torch.arange(0, L1, dtype=dtype, device=device).view(-1, 1) / L1
    #     ) * torch.pi / 2 # [L1, D]
    #     thetas[:, ::2] = 0

    #     Q_cos, Q_sen = Q * thetas.cos(), Q * thetas.sin()
    #     K_cos, K_sen = K * thetas.cos(), K * thetas.sin()

    #     return Q_cos, Q_sen, K_cos, K_sen