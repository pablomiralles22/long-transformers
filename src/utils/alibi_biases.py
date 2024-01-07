import torch
import math

from typing import Optional


class ALiBiBiases:
    __CACHE = dict()
    __REFERENCE_NUM_HEADS: int = 8

    @classmethod
    def apply(
        cls,
        seq_len: int,
        num_heads: int,
        seq_len_2: Optional[int] = None,
        base: float = 2.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        seq_len_2 = seq_len if seq_len_2 is None else seq_len_2

        if (seq_len, seq_len_2, num_heads, base) in cls.__CACHE:
            return cls.__CACHE[(seq_len, seq_len_2, num_heads, base)].to(
                device=device, dtype=dtype
            )

        seq_len_range = torch.arange(
            seq_len, device=device, dtype=dtype, requires_grad=False
        )  # [L1]
        seq_len_range_2 = torch.arange(
            seq_len_2, device=device, dtype=dtype, requires_grad=False
        )  # [L2]

        attn_mat = -torch.abs(
            seq_len_range.unsqueeze(0) - seq_len_range_2.unsqueeze(1)
        )  # [L1, L2]

        head_multiplier_exp = (
            torch.arange(num_heads, device=device, dtype=dtype, requires_grad=False)
            * cls.__REFERENCE_NUM_HEADS
            / num_heads
        )  # [H]
        head_multiplier = torch.exp(- math.log(base) * head_multiplier_exp)  # [H]

        attn_mat = attn_mat * head_multiplier.view(-1, 1, 1)  # [H, L1, L2]

        # store in cache
        cls.__CACHE[(seq_len, seq_len_2, num_heads, base)] = attn_mat

        return attn_mat


