import torch

from torch import nn

class CompressorStep(nn.Module):
    def __init__(
        self,
        d_model, vdim=None, qkdim=None,
        nhead=4,
        dropout=0.1,
        dim_feedforward=2048,
        activation_fn_cls=nn.ReLU,
        layer_norm_eps=1e-05,
    ):
        super().__init__()

        self.vdim  = d_model if vdim is None else vdim
        self.qkdim = d_model if qkdim is None else qkdim

        self.W_Q = nn.Linear(d_model, self.qkdim)
        self.W_K = nn.Linear(d_model, self.qkdim)
        self.W_V = nn.Linear(d_model, self.vdim)

        self.mh_attention = nn.MultiheadAttention(
            d_model, nhead, vdim=vdim, kdim=qkdim, batch_first=True
        )
        self.dropout_1 = nn.Dropout(dropout)
        self.layer_norm_1 = nn.LayerNorm(d_model, layer_norm_eps)
        
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            activation_fn_cls(),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.layer_norm_2 = nn.LayerNorm(d_model, layer_norm_eps)
    
    def forward(
        self,
        embeddings_1,  # [B, L1, D]; query
        embeddings_2,  # [B, L2, D]; key, value
        attention_mask=None,  # [B, L2]; key
    ):
        Q = self.W_Q(embeddings_1)
        K = self.W_K(embeddings_2)
        V = self.W_V(embeddings_2)

        if attention_mask is not None:
            attention_mask = ~attention_mask.bool()
        x, _ = self.mh_attention(Q, K, V, key_padding_mask=attention_mask)  # [B, L1, D]

        x = self.dropout_1(x)
        x = self.layer_norm_1(x + embeddings_1)

        y = self.ff(x)

        return self.layer_norm_2(x + y)


class CompressorLayer(nn.Module):
    def __init__(
        self,
        d_model, vdim=None, qkdim=None,
        nhead=4,
        dropout=0.1,
        dim_feedforward=2048,
        activation_fn_cls=nn.ReLU,
        layer_norm_eps=1e-05,
    ):
        super().__init__()

        self.compress_step = CompressorStep(
            d_model, vdim=vdim, qkdim=qkdim,
            nhead=nhead,
            dropout=dropout,
            dim_feedforward=dim_feedforward,
            activation_fn_cls=activation_fn_cls,
            layer_norm_eps=layer_norm_eps,
        )
        self.update_step = CompressorStep(
            d_model, vdim=vdim, qkdim=qkdim,
            nhead=nhead,
            dropout=dropout,
            dim_feedforward=dim_feedforward,
            activation_fn_cls=activation_fn_cls,
            layer_norm_eps=layer_norm_eps,
        )
    
    def forward(
        self,
        embeddings,  # [B, SL, D]
        memory,  # [B, ML, D]
        attention_mask=None,  # [B, SL]
    ):
        updated_mem = self.compress_step(memory, embeddings, attention_mask=attention_mask)
        updated_embeddings = self.update_step(embeddings, updated_mem)
        return updated_embeddings, updated_mem
        
class Compressor(nn.Module):
    def __init__(
        self,
        d_model, vdim=None, qkdim=None,
        num_layers=1, mem_length=128,
        nhead=4,
        dropout=0.1,
        dim_feedforward=2048,
        activation_fn_cls=nn.ReLU,
        layer_norm_eps=1e-05,
    ):
        super().__init__()

        self.compressor_layers = nn.ModuleList([
            CompressorLayer(
                d_model, vdim=vdim, qkdim=qkdim,
                nhead=nhead,
                dropout=dropout,
                dim_feedforward=dim_feedforward,
                activation_fn_cls=activation_fn_cls,
                layer_norm_eps=layer_norm_eps,
            ) for _ in range(num_layers)
        ])

        self.memory = nn.Parameter(torch.empty(mem_length, d_model))
        nn.init.xavier_uniform_(self.memory)

    
    def forward(
        self,
        embeddings,  # [B, L, D]
        attention_mask=None,  # [B, L]
        token_type_ids=None,  # [B, L]
    ):
        x, memory = embeddings, self.memory.unsqueeze(0).repeat(embeddings.shape[0], 1, 1)
        for layer in self.compressor_layers:
            x, memory = layer(x, memory, attention_mask=attention_mask)
        return x
