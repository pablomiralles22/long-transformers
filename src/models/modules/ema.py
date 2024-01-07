import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Literal

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
        self.edim = edim * dim  # = E
        self.out_dim = out_dim if out_dim is not None else dim  # = O
        self.kernel_size = kernel_size  # = K
        self.kernel_channels = (2 if direction == "bidirectional" else 1) * self.edim  # = C

        self.logit_alpha = nn.Parameter(torch.empty(self.kernel_channels))  # [C]
        self.logit_delta = nn.Parameter(torch.empty(self.kernel_channels))  # [C]
        self.beta = nn.Parameter(torch.empty(dim, self.edim))  # [D, E]
        self.eta = nn.Parameter(torch.empty(self.edim, self.out_dim))  # [E, O]

        self.reset_parameters()

        self.direction = direction

    @torch.no_grad()
    def reset_parameters(self):
        nn.init.uniform_(self.logit_alpha, -3, 3)
        nn.init.uniform_(self.logit_delta, -3, 3)
        # nn.init.xavier_normal_(self.beta)
        # nn.init.xavier_normal_(self.eta)
        nn.init.xavier_uniform_(self.beta)
        nn.init.xavier_uniform_(self.eta)

    def forward(
        self,
        embeddings,  # [B, L, D]
    ):
        (B, L, D), E = embeddings.size(), self.edim
        assert D == self.dim
        device, dtype = embeddings.device, embeddings.dtype

        # compute kernel
        alpha = torch.sigmoid(self.logit_alpha.view(-1, 1))  # [C, 1]
        delta = torch.sigmoid(self.logit_delta.view(-1, 1))  # [C, 1]

        p = alpha  # [C, 1]
        q = 1. - alpha * delta  # [C, 1]

        len_range = (
            torch
            .arange(
                self.kernel_size,
                dtype=dtype,
                device=device,
                requires_grad=False,
            )
            .view(1, -1)
        )  # [1, kernel_size]

        kernel = (p * torch.exp(torch.log(q) * len_range))  # [C, kernel_size]

        # compute output
        u = embeddings @ self.beta  # [B, L, E]
        u_T = u.transpose(-1, -2)  # [B, E, L]

        out_T = self.__convolve(u_T, kernel)  # [B, E, L]

        out = out_T.transpose(-1, -2)  # [B, L, E]
        return out @ self.eta  # [B, L, O]

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
            

        # if conv is small, apply naive alg
        if self.kernel_size < self.__FFT_THRESHOLD:
            out_T = self.__convolve_naive(inp, kernel)
        
        # else, apply FFT
        out_T = self.__convolve_fft(inp, kernel)

        return out_T[..., start:start+L]
            

    def __convolve_fft(
        self,
        inp,  # [B, E, L]
        kernel,  # [E, kernel_size]
    ):
        _, _, L = inp.shape
        _, kernel_size = kernel.shape

        fft_dim = L + kernel_size
        # get next power of 2
        fft_dim = 2 ** (fft_dim - 1).bit_length()

        inp_fft = torch.fft.rfft(inp, dim=-1, n=fft_dim)
        kernel_fft = torch.fft.rfft(kernel, dim=-1, n=fft_dim)

        # Perform element-wise multiplication in the frequency domain
        out_T_fft = inp_fft * kernel_fft

        return torch.fft.irfft(out_T_fft).type_as(inp)

    def __convolve_naive(
        self,
        inp,  # [B, E, L]
        kernel,  # [E, kernel_size]
    ):
        _, E, _ = inp.shape
        _, kernel_size = kernel.shape
        return F.conv1d(
            inp,
            kernel.flip(-1).view(E, 1, kernel_size),
            padding=kernel_size - 1,
            groups=E
        )