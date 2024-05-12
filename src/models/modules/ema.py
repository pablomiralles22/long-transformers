import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Literal
from src.models.functional.fft_conv1d import fft_conv1d

class EMA(nn.Module):
    __FFT_THRESHOLD: int = 30

    def __init__(
        self,
        dim: int,
        edim: int,
        kernel_size: int,
        out_dim: Optional[int] = None,
        direction: Literal["forward", "backward", "bidirectional"] = "forward",
    ):
        super().__init__()

        self.dim = dim  # = D
        # self.edim = edim * dim  # = E
        self.edim = edim  # = E
        self.out_dim = out_dim if out_dim is not None else dim  # = O
        self.kernel_size = kernel_size  # = K
        self.kernel_channels = (2 if direction == "bidirectional" else 1) * dim  # = C

        self.logit_alpha = nn.Parameter(torch.empty(self.kernel_channels, edim))  # [C, E]
        self.logit_delta = nn.Parameter(torch.empty(self.kernel_channels, edim))  # [C, E]
        # self.beta = nn.Parameter(torch.empty(dim, self.edim))  # [D, E]
        # self.eta = nn.Parameter(torch.empty(self.edim, self.out_dim))  # [E, O]
        self.beta = nn.Parameter(torch.empty(self.kernel_channels, edim, 1))  # [C, E, 1]
        self.eta = nn.Parameter(torch.empty(self.kernel_channels, edim))  # [C, E]

        self.reset_parameters()

        self.direction = direction

    @torch.no_grad()
    def reset_parameters(self):
        # nn.init.uniform_(self.logit_alpha, -3, 3)
        # nn.init.uniform_(self.logit_delta, -3, 3)
        nn.init.normal_(self.logit_alpha, std=0.2)
        nn.init.normal_(self.logit_delta, std=0.2)
        # nn.init.xavier_uniform_(self.beta)
        # nn.init.xavier_uniform_(self.eta)

        # beta [1, -1, 1, -1, ...] seems more stable.
        val = torch.ones(self.edim, 1)
        if self.edim > 1:
            idx = torch.tensor(list(range(1, self.edim, 2)))
            val.index_fill_(0, idx, -1.0)
        self.beta.normal_(mean=0.0, std=0.02).add_(val)
        nn.init.normal_(self.eta, mean=0.0, std=1.0)

    def forward(
        self,
        embeddings,  # [B, L, D]
    ):
        (B, L, D), E = embeddings.size(), self.edim
        assert D == self.dim
        device, dtype = embeddings.device, embeddings.dtype

        # compute kernel
        alpha = torch.sigmoid(self.logit_alpha.view(-1, self.edim, 1))  # [C, E, 1]
        delta = torch.sigmoid(self.logit_delta.view(-1, self.edim, 1))  # [C, E, 1]

        p = alpha  # [C, E, 1]
        q = 1. - alpha * delta  # [C, E, 1]

        len_range = (
            torch
            .arange(
                self.kernel_size,
                dtype=dtype,
                device=device,
                requires_grad=False,
            )
            .view(1, 1, -1)
        )  # [1, 1, kernel_size]

        kernel = ((self.beta * p) * torch.exp(torch.log(q) * len_range))  # [C, E, kernel_size]
        kernel = torch.einsum("c e k, c e -> c k", kernel, self.eta / self.edim ** 0.5)  # [C, kernel_size]

        # compute output
        u = embeddings  # [B, L, D]
        u_T = u.transpose(-1, -2)  # [B, D, L]

        out_T = self.__convolve(u_T, kernel)  # [B, D, L]

        out = out_T.transpose(-1, -2)  # [B, L, D]
        return out

    def __convolve(
        self,
        inp,  # [B, E, L]
        kernel,  # [E', kernel_size]
    ):
        _, E, L = inp.shape
        _, kernel_size = kernel.shape

        if self.direction == "backward":
            start = 0
        if self.direction == "forward":
            kernel = kernel.flip(-1)
            start = kernel_size - 1
        if self.direction == "bidirectional":
            kernel_forward, kernel_backward = torch.split(kernel, [E, E], dim=0)
            kernel = (
                F.pad(kernel_backward, (kernel_size - 1, 0)) + 
                F.pad(kernel_forward.flip(-1), (0, kernel_size - 1))
            )  # [E, 2 * kernel_size - 1]
            start = kernel_size - 1
            

        if self.kernel_size < self.__FFT_THRESHOLD: # if conv is small, apply naive alg
            out_T = self.__convolve_naive(inp, kernel)
        else: # else, apply FFT
            out_T = fft_conv1d(inp, kernel)

        return out_T[..., start:start+L]
            

    def __convolve_naive(
        self,
        inp,  # [B, E, L]
        kernel,  # [E, kernel_size]
    ):
        _, E, _ = inp.shape
        _, kernel_size = kernel.shape
        kernel = kernel.flip(-1).view(E, 1, kernel_size)
        return F.conv1d(inp, kernel, padding=kernel_size - 1, groups=E)