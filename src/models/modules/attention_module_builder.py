from torch import nn
from typing import Literal
from src.models.modules.aft import AFT
from src.models.modules.alibi_multihead_attention import ALiBiMultiheadAttention
from src.models.modules.ema_multihead_attention import EMAMultiheadAttention
from src.models.modules.positional_bias_multihead_attention import PositionalBiasMultiheadAttention
from src.models.modules.rotary_multihead_attention import RotaryMultiheadAttention
from src.models.modules.multihead_attention import MultiheadAttention
from src.models.modules.compress_multihead_attention import CompressMultiheadAttention
from src.models.modules.positive_multihead_attention import PositiveMultiheadAttention

AttentionModuleType = Literal[
    "std",
    "positional_bias",
    "rotary",
    "alibi",
    "ema",
    "aft",
    "compress",
    "positive",
]

class AttentionModuleBuilder:
    @classmethod
    def build(cls, d_model: int, nhead: int, params: dict) -> nn.Module:
        type_str: AttentionModuleType = params.pop("type")

        params["d_model"] = d_model
        params["nhead"] = nhead

        match type_str:
            case "std":
                return MultiheadAttention(**params)
            case "positional_bias":
                return PositionalBiasMultiheadAttention(**params)
            case "rotary":
                return RotaryMultiheadAttention(**params)
            case "alibi":
                return ALiBiMultiheadAttention(**params)
            case "ema":
                return EMAMultiheadAttention(**params)
            case "aft":
                return AFT(**params)
            case "compress":
                return CompressMultiheadAttention(**params)
            case "positive":
                return PositiveMultiheadAttention(**params)
            case _:
                raise ValueError(f"Unknown attention module type: {type_str}")
