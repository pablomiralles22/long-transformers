from src.models.conv_transformer import ConvTransformer
from src.models.transformer_base import TransformerModel


class ModelBuilder:
    @classmethod
    def build_model(cls, model_name, model_params, embedding_params):
        match model_name:
            case "conv-transformer":
                return cls._build_conv_transformer(model_params, embedding_params)
            case "transformer":
                return cls._build_transformer(model_params, embedding_params)
            case _:
                raise ValueError(f"Unknown model: {model_name}")

    @classmethod
    def _build_conv_transformer(cls, model_params, embedding_params):
        embedding_params["embedding_dim"] = (
            model_params["conv_layers_params"][0]["conv_params"]["in_channels"]
        )
        return ConvTransformer(
            embedding_params,
            model_params["conv_layers_params"],
            model_params["transformer_params"],
        )

    @classmethod
    def _build_transformer(cls, model_params, embedding_params):
        embedding_params["embedding_dim"] = model_params["layer_params"]["d_model"]
        return TransformerModel(
            embedding_params,
            model_params["layer_params"],
            model_params["num_layers"],
        )