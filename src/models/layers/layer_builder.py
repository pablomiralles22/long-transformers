from torch import nn
from typing import Literal

from src.models.layers.residual_block import ResidualBlock
from src.models.layers.positional_encoding_layer import PositionalEncodingLayer
from src.models.layers.layer import Layer, Overlayer, ConvLayer
from src.models.layers.transformer_encoder_layer import TransformerEncoderLayer

LayerType = Literal[
    "linear",
    "conv",
    "max_pool",
    "mean_pool",
    "relu",
    "dropout",
    "batch_norm",
    "layer_norm",
    "residual_block",
    "positional_encoding",
    "transformer_encoder_layer",
]

class LayerBuilder:
    @classmethod
    def build(cls, name: LayerType, params: dict) -> Layer:
        match name:
            case "linear":
                return Overlayer(nn.Linear(**params))
            # conv and pooling
            case "conv":
                return ConvLayer(nn.Conv1d(**params))
            case "max_pool":
                return ConvLayer(nn.MaxPool1d(**params))
            case "mean_pool":
                return ConvLayer(nn.AvgPool1d(**params))
            # activations
            case "relu":
                return Overlayer(nn.ReLU())
            # regularization
            case "dropout":
                return Overlayer(nn.Dropout(**params))
            # normalization
            case "batch_norm":
                return Overlayer(nn.BatchNorm1d(**params))
            case "layer_norm":
                return Overlayer(nn.LayerNorm(**params))
            # other
            case "residual_block":
                layers = [
                    cls.build(subname, subparams)
                    for subname, subparams in params["layer_params"]
                ]
                return ResidualBlock(layers)
            case "positional_encoding":
                return PositionalEncodingLayer()
            case "transformer_encoder_layer":
                return TransformerEncoderLayer(**params)
            case _:
                raise ValueError(f"Invalid LayerType name: {name}")
