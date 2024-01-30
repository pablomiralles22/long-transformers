import torch
import math

class RotaryEmbedding:
    __CACHE = dict()

    @classmethod
    def apply(
        cls,
        X,  # [B, H, L, D]
        thetas=None,  # [D // 2] or [H, D // 2] or None
        freq=10000,
    ):
        *B, H, L, D = X.shape
        assert D % 2 == 0, "Error: the embedding dimension should be divisible by 2"

        device, dtype = X.device, X.dtype

        rotation_mat = cls.__build_rotation_matrix(thetas, L, D, freq, device=device, dtype=dtype) # [H, L, D // 2, 2, 2] or [1, L, D // 2, 2, 2]
        X_reshaped = X.view(*B, H, L, D // 2, 1, 2) # (...B, H, L, D // 2, 1, 2)

        return (
            (X_reshaped * rotation_mat) # (...B, L, D // 2, 2, 2)
            .sum(dim=-1) # (...B, L, D // 2, 2)
            .view(*B, H, L, D) # (...B, L, D)
        )
    
    @classmethod
    def __build_rotation_matrix(cls, thetas, L, D, freq, device, dtype):
        should_cache = thetas is None
        # try to retrieve from cache
        if should_cache is True and (L, D, freq) in cls.__CACHE:
            return cls.__CACHE[(L, D, freq)]

        lengths = torch.arange(0, L, requires_grad=False, device=device, dtype=dtype) # (L)

        if thetas is None:
            thetas_inds = torch.arange(0, D // 2, requires_grad=False, device=device, dtype=dtype) # (D // 2)
            thetas = torch.exp(-2 * math.log(freq) * (thetas_inds // 2) / D) # (D // 2)
        else:
            thetas = thetas.unsqueeze(-2) / L  # [H, 1, D // 2] or [1, D // 2]

        prod = torch.einsum("a , ...b -> ...ab", lengths, thetas) # [L, D // 2] or [H, L, D // 2]
        cosines = torch.cos(prod) # [L, D // 2] or [H, L, D // 2]
        sines = torch.sin(prod) # [L, D // 2] or [H, L, D // 2]

        rotation_mat = (
            torch
            .stack([cosines, -sines, sines, cosines], dim=-1)
            .view(-1, L, D // 2, 2, 2)
        ) # [H, L, D // 2, 2, 2] or [1, L, D // 2, 2, 2]

        # save to cache
        if should_cache is True:
            cls.__CACHE[(L, D, freq)] = rotation_mat

        return rotation_mat