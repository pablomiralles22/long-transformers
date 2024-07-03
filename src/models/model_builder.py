from copy import deepcopy
from typing import Literal

from src.models.model_with_embedding import ModelWithEmbedding
from src.models.layered_model import LayeredModel
from src.models.layers.layer_builder import LayerBuilder

ModelType = Literal[
    "layered",
]

class ModelBuilder:
    @classmethod
    def build_model(
        cls,
        type: str,
        params: dict,
        vocab_size: int,
        padding_idx: int,
    ) -> ModelWithEmbedding:
        match type:
            case "layered":
                model = cls._build_layered_model(params)
            case "single_layer":
                model = cls._build_single_layer_model(params)
            case _:
                raise ValueError(f"Unknown model: {type}")
        return ModelWithEmbedding(model, vocab_size, padding_idx)

    @classmethod
    def _build_layered_model(cls, params: dict) -> LayeredModel:
        params = deepcopy(params)
        layers = params.pop("layers")
        layers = [
            LayerBuilder.build(layer_name, layer_params)
            for layer_name, layer_params in layers
        ]
        return LayeredModel(**params, layers=layers)

    @classmethod
    def _build_single_layer_model(cls, params: dict) -> LayeredModel:
        params = deepcopy(params)

        num_layers = params.pop("num_layers")
        layer = params.pop("layer")
        layer_type = layer.pop("type")
        layer_params = layer.pop("params")

        layer = LayerBuilder.build(layer_type, layer_params)
        layers = [deepcopy(layer) for _ in range(num_layers)]
        return LayeredModel(**params, layers=layers)