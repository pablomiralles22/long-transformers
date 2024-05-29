import torch

def log_positional_embedding(
    idxs: torch.Tensor,  # [L]
    d_model: int,
    device: torch.device = None,
    dtype: torch.dtype = torch.float,
):
    L = idxs.shape[0]
    idxs = idxs.unsqueeze(1).type(dtype)  # [L, 1]

    mult_term = torch.arange(0, d_model, dtype=dtype, device=device, requires_grad=False) / d_model  # [D]
    pe = idxs / L * mult_term  # [L, D]

    return pe