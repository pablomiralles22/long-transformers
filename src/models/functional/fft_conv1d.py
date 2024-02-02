import torch

def fft_conv1d(
    inp,  # [B, D, L]
    kernel,  # [D, K]
):
    B, D, L = inp.shape
    D, K = kernel.shape

    fft_dim = L + K
    # get next power of 2
    fft_dim = 2 ** (fft_dim - 1).bit_length()

    inp_fft = torch.fft.rfft(inp, dim=-1, n=fft_dim)
    kernel_fft = torch.fft.rfft(kernel, dim=-1, n=fft_dim)

    # Perform element-wise multiplication in the frequency domain
    out_T_fft = inp_fft * kernel_fft

    return torch.fft.irfft(out_T_fft, dim=-1, n=fft_dim).type_as(inp)
