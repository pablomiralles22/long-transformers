from torch import nn

from src.models.layers.layer import Layer

class OneSidedConv(Layer):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        groups: int = 1,
    ):
        super(OneSidedConv, self).__init__()
        self.kernel_size = kernel_size

        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            groups=groups,
            padding=kernel_size - 1,
        )

    def forward(
        self,
        embeddings,  # [B, L, D]
        attention_mask=None,  # [B, L]
        token_type_ids=None,  # [B, L]
    ):
        *_, L, _ = embeddings.shape
        # embeddings_skip = embeddings

        embeddings = embeddings * attention_mask.unsqueeze(-1)  # [B, L, D]
        embeddings = embeddings.transpose(-1, -2)  # [B, D, L]
        embeddings = self.conv(embeddings)  # [B, D, L+K-1]
        embeddings = embeddings[..., :L].transpose(-1, -2)

        # if embeddings.shape == embeddings_skip.shape:
        #     return embeddings + embeddings_skip
        return embeddings
