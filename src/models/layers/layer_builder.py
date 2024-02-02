from torch import nn
from typing import Literal

from src.models.layers.residual_block import ResidualBlock
from src.models.layers.layer import Layer, Overlayer, ConvLayer, UnpackLayer
from src.models.layers.transformer_encoder_layer import TransformerEncoderLayer
from src.models.layers.local_transformer_encoder_layer import LocalTransformerEncoderLayer
# from src.models.layers.conv_transformer_encoder_layer import ConvTransformerEncoderLayer
from src.models.layers.embeddings import PositionalEmbeddingLayer, TokenTypeEmbeddingLayer
from src.models.layers.compressor_layer import Compressor
from src.models.layers.one_sided_conv import OneSidedConv
from src.models.modules.ema import EMA
from src.models.layers.gmlp import GMLP

LayerType = Literal[
    "linear",

    "conv",
    "one_sided_conv",
    "max_pool",
    "mean_pool",

    "relu",
    "dropout",

    "batch_norm",
    "layer_norm",

    "residual_block",

    "positional_embedding",
    "token_type_embedding",

    "transformer_encoder_layer",
    "local_transformer_encoder_layer",
    "conv_transformer_encoder_layer",
    "compressor",
    "gmlp"

    "lstm",
    "rnn",
    "ema",
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
            case "one_sided_conv":
                return OneSidedConv(**params)
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
            case "residual_block":
                layers = [
                    cls.build(subname, subparams)
                    for subname, subparams in params["layer_params"]
                ]
                return ResidualBlock(layers)
            # embeddings
            case "positional_embedding":
                return PositionalEmbeddingLayer(**params)
            case "token_type_embedding":
                return TokenTypeEmbeddingLayer(**params)
            # transformers encoder layers
            case "transformer_encoder_layer":
                return TransformerEncoderLayer(**params)
            # case "conv_transformer_encoder_layer":
            #     return ConvTransformerEncoderLayer(**params)
            case "local_transformer_encoder_layer":
                return LocalTransformerEncoderLayer(**params)
            case "compressor":
                return Compressor(**params)
            case "gmlp":
                return GMLP(**params)
            # lstm
            case "lstm":
                return UnpackLayer(nn.LSTM(**params))
            case "rnn":
                return UnpackLayer(nn.RNN(**params))
            # ema
            case "ema":
                return Overlayer(EMA(**params))
            case _:
                raise ValueError(f"Invalid LayerType name: {name}")
