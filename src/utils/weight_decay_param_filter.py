class WeightDecayParamFilter:
    __WHITELISTED_WORDS = ("linear", "conv", "kernel", "weight")
    __BLACKLISTED_WORDS = ("bias", "norm", "embedding")

    @classmethod
    def filter(cls, model):
        weight_decay_params = []
        no_weight_decay_params = []

        for name, param in model.named_parameters():
            if any(word in name for word in cls.__BLACKLISTED_WORDS):
                no_weight_decay_params.append(param)
            elif any(word in name for word in cls.__WHITELISTED_WORDS):
                weight_decay_params.append(param)
            else:
                no_weight_decay_params.append(param)

        return weight_decay_params, no_weight_decay_params