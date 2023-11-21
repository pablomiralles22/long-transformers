from torch import nn
from src.models.positional_encoding import positional_encoding


class ConvTransformerInterleavedLayer(nn.Module):
    def __init__(
        self,
        conv_params: dict,
        dim_feedforward: int,
        dropout_params: dict,
        transformer_layer_params: dict,
    ):
        super(ConvTransformerInterleavedLayer, self).__init__()

        self.conv = nn.Conv1d(**conv_params)
        self.ff = nn.Sequential(
            nn.Linear(conv_params["out_channels"], dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, conv_params["out_channels"]),
        )
        self.layer_norm1 = nn.LayerNorm(conv_params["out_channels"])
        self.layer_norm2 = nn.LayerNorm(conv_params["out_channels"])
        self.dropout = nn.Dropout(**dropout_params)
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
            **transformer_layer_params,
            batch_first=True,
        )

    def forward(
        self,
        x,  # (...BATCH, LENGTH, EMBED_DIM)
        attention_mask,  # (...BATCH, LENGTH)
    ):
        x = x.transpose(-1, -2)  # (...BATCH, EMBED_DIM, LENGTH)
        x = self.conv(x)  # (...BATCH, NEW_EMBED_DIM, LENGTH)
        x = x.transpose(-1, -2)  # (...BATCH, LENGTH, NEW_EMBED_DIM)
        x = self.layer_norm1(x)  # (...BATCH, LENGTH, NEW_EMBED_DIM)
        x = self.ff(x)  # (...BATCH, LENGTH, NEW_EMBED_DIM)
        x = self.layer_norm2(x)  # (...BATCH, LENGTH, NEW_EMBED_DIM)
        x = self.transformer_encoder_layer(x, src_key_padding_mask=attention_mask)   # (...BATCH, LENGTH, NEW_EMBED_DIM)
        return x


class ConvTransformerInterleaved(nn.Module):
    def __init__(
        self,
        embedding_params: dict, # TODO
        layers_params: list[dict],
    ):
        super(ConvTransformerInterleaved, self).__init__()

        self.embedding = nn.Embedding(**embedding_params)

        # Convolutional layers
        self.layers = nn.ModuleList()
        for layer_params in layers_params:
            self.layers.append(
                ConvTransformerInterleavedLayer(
                    layer_params["conv_params"],
                    layer_params["dim_feedforward"],
                    layer_params["dropout_params"],
                    layer_params["transformer_layer_params"],
                )
            )


    def forward(
        self,
        input_ids,  # (...BATCH, LENGTH)
        src_key_padding_mask=None,  # (...BATCH, LENGTH)
    ):
        x = self.embedding(input_ids)  # (...BATCH, LENGTH, EMBED_DIM)
        *_, length, embed_dim = x.shape
        device = x.device
        x += positional_encoding(length, embed_dim).to(device)

        for layer in self.layers:
            x = layer(x, src_key_padding_mask=src_key_padding_mask)

        return x
