from torch import nn
from typing import Literal
from src.models.modules.alibi_multihead_attention import ALiBiMultiheadAttention
from src.models.modules.ema_multihead_attention import EMAMultiheadAttention
from src.models.modules.positional_bias_multihead_attention import PositionalBiasMultiheadAttention
from src.models.modules.rotary_multihead_attention import RotaryMultiheadAttention
from src.models.modules.multihead_attention import MultiheadAttention

AttentionModuleType = Literal[
    "std",
    "positional_bias",
    "rotary",
    "alibi",
    "ema",
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
            case _:
                raise ValueError(f"Unknown attention module type: {type_str}")
