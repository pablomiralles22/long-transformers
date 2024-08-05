from torch import optim
from src.utils.weight_decay_param_filter import WeightDecayParamFilter

class OptimizerBuilder:
    __STR_TO_OPTIMIZER_CLS = {
        'adamw': optim.AdamW,
        'adam': optim.Adam,
        'sgd': optim.SGD,
    }

    @classmethod
    def build(cls, model, optimizer_config):
        optimizer_name = optimizer_config.pop("_name_")
        optimizer_cls = cls.__STR_TO_OPTIMIZER_CLS[optimizer_name]
        
        parameters = model.parameters()

        if optimizer_name.startswith("adam"):
            weight_decay = optimizer_config.get("weight_decay", 0.01)
            if weight_decay > 0:
                # separate parameters with weight decay and without weight decay
                weight_decay_params, no_weight_decay_params = WeightDecayParamFilter.filter(model, debug=False)
                optimizer_config["weight_decay"] = 0.0

                parameters = [
                    {"params": weight_decay_params, "weight_decay": weight_decay},
                    {"params": no_weight_decay_params, "weight_decay": 0.0},
                ]

        return optimizer_cls(parameters, **optimizer_config)