import torch
import math

def hyperbolic_positional_embedding(
    idxs: torch.Tensor,  # [L]
    d_model: int,
    device: torch.device = None,
    dtype: torch.dtype = torch.float,
):
    if d_model % 2 != 0:
        raise ValueError(
            "Cannot use sinh/cosh positional encoding with "
            f"odd dim (got dim={d_model})"
        )
    L = idxs.shape[0]

    idxs = idxs.unsqueeze(1).type(dtype)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=dtype, device=device, requires_grad=False)
        *
        (-math.log(10000.0) / d_model)
    )

    pe = torch.empty(L, d_model, device=device, requires_grad=False)
    pe[:, 0::2] = torch.sinh(idxs.float() / L * div_term)
    pe[:, 1::2] = torch.cosh(idxs.float() / L * div_term)

    return pe