import torch

def log_positional_embedding(
    idxs: torch.Tensor,  # [L]
    d_model: int,
    device: torch.device = None,
    dtype: torch.dtype = torch.float,
):
    idxs = idxs.unsqueeze(1).type(dtype)  # [L, 1]

    mult_term = torch.arange(0, d_model, dtype=dtype, device=device, requires_grad=False) / d_model  # [D]
    mult_term[::2] = 0.
    pe = 1e-3 * idxs * mult_term  # [L, D]

    return pe