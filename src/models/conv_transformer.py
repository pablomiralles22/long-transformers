from torch import nn
from utils import positional_encoding


class ConvLayer(nn.Module):
    def __init__(
        self,
        conv_params: dict,
        dropout_params: dict,
    ):
        super(ConvLayer, self).__init__()

        self.conv = nn.Conv1d(**conv_params)
        self.ff = nn.Sequential(
            nn.Linear(conv_params["out_channels"], 2 * conv_params["out_channels"]),
            nn.ReLU(),
            nn.Linear(2 * conv_params["out_channels"], conv_params["out_channels"]),
        )
        self.layer_norm1 = nn.LayerNorm(conv_params["out_channels"])
        self.layer_norm2 = nn.LayerNorm(conv_params["out_channels"])
        self.dropout = nn.Dropout(**dropout_params)

    def forward(self, x):
        x = self.conv(x)
        x = x.transpose(-1, -2)
        x = self.layer_norm1(x)
        x = self.ff(x)
        x = self.layer_norm2(x)
        x = x.transpose(-1, -2)
        return x


class ConvTransformer(nn.Module):
    def __init__(
        self,
        embedding_params: dict,
        conv_layers_params: list[dict],
        transformer_params: dict,
    ):
        super(ConvTransformer, self).__init__()

        self.embedding = nn.Embedding(**embedding_params)

        # Convolutional layers
        self.conv_layers = nn.ModuleList()
        for conv_layer_params in conv_layers_params:
            self.conv_layers.append(
                ConvLayer(
                    conv_layer_params["conv_params"],
                    conv_layer_params["dropout_params"],
                )
            )

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(**transformer_params["layer_params"])
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=transformer_params["num_layers"]
        )

    def forward(
        self,
        input_ids,  # (...BATCH, LENGTH)
        src_key_padding_mask=None,  # (...BATCH, LENGTH)
    ):
        x = self.embedding(input_ids)  # (...BATCH, LENGTH, EMBED_DIM)
        x = x.transpose(-1, -2)  # (...BATCH, EMBED_DIM, LENGTH)
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        x = x.transpose(-1, -2)  # (...BATCH, LENGTH, D_TRANSFORMER)
        *_, length, d_model = x.shape
        device = x.device
        x += positional_encoding(length, d_model).to(device)
        x = self.transformer_encoder.forward(
            x, src_key_padding_mask=src_key_padding_mask
        )
        return x
