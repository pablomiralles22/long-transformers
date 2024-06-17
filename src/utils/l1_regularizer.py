import torch

@torch.no_grad()
def l1_regularization_hook(param, l1_lambda=0.01):
    def hook(grad):
        return grad + l1_lambda * torch.sign(param)
    return hook

class L1Regularizer:
    __WHITELISTED_WORDS = ("linear", "conv", "kernel", "weight")
    __BLACKLISTED_WORDS = ("bias", "norm", "embedding")

    @classmethod
    def apply(cls, model, l1_lambda=0.01, debug=False):
        l1_params = []
        no_l1_params = []

        l1_names = []
        no_l1_names = []

        for name, param in model.named_parameters():
            if any(word in name for word in cls.__BLACKLISTED_WORDS):
                no_l1_params.append(param)
                no_l1_names.append(name)
            elif any(word in name for word in cls.__WHITELISTED_WORDS):
                l1_params.append(param)
                l1_names.append(name)
            else:
                no_l1_params.append(param)
                no_l1_names.append(name)

        if debug is True:
            print("Adding l1 reg")
            for name in l1_names:
                print(f"\t{name}")
            print()
            print("Skipping l1 reg")
            for name in no_l1_names:
                print(f"\t{name}")

        for param in l1_params:
            param.register_hook(l1_regularization_hook(param, l1_lambda))