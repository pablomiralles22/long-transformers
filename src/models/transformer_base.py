from utils import positional_encoding
from torch import nn

class TransformerModel(nn.Module):
    def __init__(
        self,
        embedding_params: dict,
        transformer_encoder_params: dict,
        num_transformer_encoder_layers: int
    ):
        super(TransformerModel, self).__init__()

        self.embedding = nn.Embedding(**embedding_params)
        encoder_layer = nn.TransformerEncoderLayer(**transformer_encoder_params)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_encoder_layers)

    def forward(
        self,
        input_ids,  # (...BATCH, LENGTH)
        src_key_padding_mask=None,  # (...BATCH, LENGTH)
    ):
        x = self.embedding(input_ids)  # (...BATCH, LENGTH, EMBED_DIM)
        *_, length, d_model = x.shape
        device = x.device
        x += positional_encoding(length, d_model).to(device)
        x = self.transformer_encoder.forward(x, src_key_padding_mask=src_key_padding_mask)
        return x