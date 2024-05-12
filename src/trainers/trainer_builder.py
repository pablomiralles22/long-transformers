from src.trainers.cifar10_trainer import CIFAR10Module
from src.trainers.pathfinder_trainer import PathfinderModule
from src.trainers.listops_trainer import ListopsModule
from src.trainers.text_classification_trainer import TextClassificationModule
from src.trainers.mlm_trainer import MLMModule


class TrainerBuilder:
    @classmethod
    def build_trainer(
        cls,
        model_params: dict,
        data_module_params: dict,
        head_params: dict,
        optimizer_params: dict,
    ):
        name: str = data_module_params["name"]
        match name:
            case "text-classification":
                return TextClassificationModule(
                    model_params=model_params,
                    data_module_params=data_module_params,
                    head_params=head_params,
                    optimizer_params=optimizer_params,
                )
            case "text-retrieval":
                return TextClassificationModule(
                    model_params=model_params,
                    data_module_params=data_module_params,
                    head_params=head_params,
                    optimizer_params=optimizer_params,
                )
            case "listops":
                return ListopsModule(
                    model_params=model_params,
                    data_module_params=data_module_params,
                    head_params=head_params,
                    optimizer_params=optimizer_params,
                )
            case "cifar10":
                return CIFAR10Module(
                    model_params=model_params,
                    data_module_params=data_module_params,
                    head_params=head_params,
                    optimizer_params=optimizer_params,
                )
            case "pathfinder":
                return PathfinderModule(
                    model_params=model_params,
                    data_module_params=data_module_params,
                    head_params=head_params,
                    optimizer_params=optimizer_params,
                )
            case "mlm":
                return MLMModule(
                    model_params=model_params,
                    data_module_params=data_module_params,
                    head_params=head_params,
                    optimizer_params=optimizer_params,
                )
            case _:
                raise ValueError(f"Unknown task: {name}")
