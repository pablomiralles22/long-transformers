from torch import nn
from src.models.layers.rotary_transformer_encoder_layer import RotaryMultiheadSelfAttention


class ConvTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        vdim=None,
        qkdim=None,
        nhead=4,
        dropout=0.1,
        dim_feedforward=2048,
        activation_fn_cls=nn.ReLU,
        layer_norm_eps=1e-05,
        norm_first=True,
        conv_out_channels=None,
        conv_kernel_size=3,
        conv_groups=1,
    ):
        super().__init__()

        self.norm_first = norm_first

        conv_out_channels = conv_out_channels or d_model

        self.conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=conv_out_channels,
            kernel_size=conv_kernel_size,
            padding="same",
            groups=conv_groups,
        )
        self.dropout_0 = nn.Dropout(dropout)
        self.layer_norm_0 = nn.LayerNorm(conv_out_channels, layer_norm_eps)

        self.mh_attention = RotaryMultiheadSelfAttention(conv_out_channels, nhead)
        self.dropout_1 = nn.Dropout(dropout)
        self.layer_norm_1 = nn.LayerNorm(conv_out_channels, layer_norm_eps)
        
        self.ff = nn.Sequential(
            nn.Linear(conv_out_channels, dim_feedforward),
            activation_fn_cls(),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.layer_norm_2 = nn.LayerNorm(d_model, layer_norm_eps)
    
    def forward(
        self,
        embeddings,  # (BATCH, LENGTH, d_model)
        attention_mask=None,  # (BATCH, LENGTH)
        token_type_ids=None,  # (BATCH, LENGTH)
    ):
        embeddings = embeddings * attention_mask.unsqueeze(-1).float()  # set padding to 0
        x = self.conv(embeddings.transpose(-1, -2)).transpose(-1, -2)
        x = self.dropout_0(x)
        x = self.layer_norm_0(x)

        if self.norm_first is True:
            # y = self.layer_norm_1(x)
            y = self.mh_attention(x, key_attention_mask=attention_mask)
            y = embeddings + self.dropout_1(y)
            return y + self.ff(self.layer_norm_2(y))

        y = self.mh_attention(x, attention_mask)
        y = self.layer_norm_1(embeddings + self.dropout_1(y))
        return self.layer_norm_2(y + self.ff(y))