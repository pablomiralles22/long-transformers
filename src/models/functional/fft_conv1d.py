import torch

def fft_conv1d(
    inp,  # [B, D, L]
    kernel,  # [N, K]
):
    B, D, L = inp.shape
    N, K = kernel.shape

    assert D % N == 0, "num_channels must be divisible by num_kernels"

    fft_dim = L + K
    # get next power of 2, it might be faster
    fft_dim = 2 ** (fft_dim - 1).bit_length()

    inp_fft = torch.fft.rfft(inp, dim=-1, n=fft_dim, norm="ortho")
    kernel_fft = torch.fft.rfft(kernel, dim=-1, n=fft_dim, norm="ortho")

    if 1 < N < D:
        kernel_fft = kernel_fft.repeat(D // N, 1)

    # Perform element-wise multiplication in the frequency domain
    out_fft = inp_fft * kernel_fft

    return torch.fft.irfft(out_fft, dim=-1, n=fft_dim, norm="ortho").type_as(inp)
