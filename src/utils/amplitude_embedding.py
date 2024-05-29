import torch

class AmplitudeEmbedding:
    __CACHE = dict()

    @classmethod
    def apply(cls, Q, K, base_amp=1e-3):
        # Q: [*, L, D]
        # K: [*, L, D]
        *B, L, D = Q.shape
        device, dtype = Q.device, Q.dtype

        amp_mat_Q, amp_mat_K = cls.__build_amp_matrix(L, D, base_amp, device=device, dtype=dtype) #  [L, D]

        return Q * amp_mat_Q, K * amp_mat_K
    
    @classmethod
    def __build_amp_matrix(cls, L, D, base_amp, device, dtype):
        # try to retrieve from cache
        if (L, D, base_amp) in cls.__CACHE:
            return cls.__CACHE[(L, D, base_amp)]

        lengths = torch.arange(0, L, requires_grad=False, device=device, dtype=dtype) / L # (L)

        amps_inds = torch.arange(0, D, requires_grad=False, device=device, dtype=dtype) / D # (D)
        amps = base_amp * amps_inds / D # (D)
        amps[::2] = 0

        prod = torch.einsum("a , b -> ab", lengths, amps) # [L, D]
        amp_mat_Q = torch.exp(prod)  # [L, D]
        amp_mat_K = torch.exp(-prod)

        cls.__CACHE[(L, D, base_amp)] = (amp_mat_Q, amp_mat_K)
        return amp_mat_Q, amp_mat_K