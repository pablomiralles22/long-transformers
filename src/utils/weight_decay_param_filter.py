class WeightDecayParamFilter:
    __WHITELISTED_WORDS = ("linear", "conv", "kernel", "weight")
    __BLACKLISTED_WORDS = ("bias", "norm", "embedding")

    @classmethod
    def filter(cls, model, debug=False):
        weight_decay_params = []
        no_weight_decay_params = []

        weight_decay_names = []
        no_weight_decay_names = []

        for name, param in model.named_parameters():
            if any(word in name for word in cls.__BLACKLISTED_WORDS):
                no_weight_decay_params.append(param)
                no_weight_decay_names.append(name)
            elif any(word in name for word in cls.__WHITELISTED_WORDS):
                weight_decay_params.append(param)
                weight_decay_names.append(name)
            else:
                no_weight_decay_params.append(param)
                no_weight_decay_names.append(name)

        if debug is True:
            print("Adding weight decay")
            for name in weight_decay_names:
                print(f"\t{name}")
            print()
            print("Skipping weight decay")
            for name in no_weight_decay_names:
                print(f"\t{name}")


        return weight_decay_params, no_weight_decay_params