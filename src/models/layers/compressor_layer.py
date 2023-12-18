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

        self.vdim  = self.d_model if vdim is None else vdim
        self.qkdim = self.d_model if qkdim is None else qkdim

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

        Q = self.__separate_heads(Q)  # [B, H, L1, qkdim]
        K = self.__separate_heads(K)  # [B, H, L2, qkdim]
        V = self.__separate_heads(V)  # [B, H, L2, vdim]

        x, _ = self.mh_attention(Q, K, V, key_padding_mask=attention_mask)  # [B, H, L1, vdim]
        x = self.__join_heads(x)  # [B, L1, d_model]

        x = self.dropout_1(x)
        x = self.layer_norm_1(x + embeddings_1)

        y = self.ff(x)

        return self.layer_norm_2(x + y)
    
    def __separate_heads(self, mat):
        # [B, L, D] -> [B, H, L, D/H]
        *B, L, D = mat.shape
        return mat.view(*B, L, self.nhead, D // self.nhead).transpose(-2, -3)
    
    def __join_heads(self, mat):
        # [B, H, L, D/H] -> [B, L, D]
        *B, H, L, DH = mat.shape
        return mat.transpose(-2, -3).contiguous().view(*B, L, H * DH)


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
        
