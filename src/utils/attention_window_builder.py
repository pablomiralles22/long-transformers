import torch

from typing import Optional

class AttentionWindowBuilder:
    __CACHE = dict()

    @classmethod
    def build_idxs(
        cls,
        window: int,
        length: int,
        device: Optional[torch.device] = None
    ):
        if (window, length) in cls.__CACHE:
            return cls.__CACHE[(window, length)]

        offset_start = -window // 2 + 1
        offset_end = window // 2 + 1
        offsets = (
            torch.arange(offset_start, offset_end, requires_grad=False, device=device)
            .view(1, -1)
        )

        min_idx = -offset_start
        max_idx = length - offset_end
        start_idxs = (
            torch.arange(0, length, requires_grad=False, device=device)
            .clip(min_idx, max_idx)
            .view(-1, 1)
        )
        idxs = start_idxs + offsets

        return idxs

    @classmethod
    def build_attention_mask(
        cls,
        window: int,
        length: int,
        padding_mask: torch.Tensor,  # [B, L]
        device: Optional[torch.device] = None,
    ):  # -> [B, L, W]
        idxs = cls.build_idxs(window, length, device=device).unsqueeze(0)   # [1, L, W]
        max_lengths = padding_mask.sum(dim=-1, keepdim=True).unsqueeze(-1)  # [B, 1, 1]
        attention_mask = idxs < max_lengths  # [B, L, W]
        attention_mask.requires_grad_(False)
        return attention_mask