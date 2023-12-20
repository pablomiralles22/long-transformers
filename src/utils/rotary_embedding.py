import torch
import math

class RotaryEmbedding:
    __CACHE = dict()

    @classmethod
    def apply(cls, X, freq=10000):
        # X: (BATCH, SEQ_LEN, d_model)
        *batch_size, seq_len, d_model = X.shape
        assert d_model % 2 == 0, "Error: the embedding dimension should be divisible by 2"

        device, dtype = X.device, X.dtype

        rotation_mat = cls.__build_rotation_matrix(seq_len, d_model, freq, device=device, dtype=dtype) # (seq_len, d_model // 2, 2, 2)
        X_reshaped = X.view(*batch_size, seq_len, d_model // 2, 1, 2) # (...batch, seq_len, d_model // 2, 1, 2)

        return (
            (X_reshaped * rotation_mat) # (...batch, seq_len, d_model // 2, 2, 2)
            .sum(dim=-1) # (...batch, seq_len, d_model // 2, 2)
            .view(*batch_size, seq_len, d_model) # (...batch, seq_len, d_model)
        )
    
    @classmethod
    def __build_rotation_matrix(cls, seq_len, d_model, freq, device, dtype):
        # try to retrieve from cache
        if (seq_len, d_model, freq) in cls.__CACHE:
            return cls.__CACHE[(seq_len, d_model, freq)]

        lengths = torch.arange(0, seq_len, requires_grad=False, device=device, dtype=dtype) # (seq_len)
        thetas_inds = torch.arange(0, d_model // 2, requires_grad=False, device=device, dtype=dtype) # (d_model // 2)
        thetas = torch.exp(-2 * math.log(freq) * (thetas_inds // 2) / d_model) # (d_model // 2)

        prod = torch.einsum("a , b -> ab", lengths, thetas) # (seq_len, d_model // 2)
        cosines = torch.cos(prod) # (seq_len, d_model // 2)
        sines = torch.sin(prod) # (seq_len, d_model // 2)

        rotation_mat = (
            torch
            .stack([cosines, -sines, sines, cosines], dim=-1)
            .view(seq_len, d_model // 2, 2, 2)
        ) # (seq_len, d_model // 2, 2, 2)

        # save to cache
        cls.__CACHE[(seq_len, d_model, freq)] = rotation_mat

        return rotation_mat