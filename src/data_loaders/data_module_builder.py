from src.data_loaders.pathfinder import PathfinderDataModule
from src.data_loaders.text_classification import TextClassificationDataModule
from src.data_loaders.listops import ListopsDataModule
from src.data_loaders.cifar10 import CIFAR10DataModule
from src.data_loaders.text_retrieval import TextRetrievalDataModule


class DataModuleBuilder:
    @classmethod
    def build_data_module(cls, name: str, params: dict):
        match name:
            case "text-classification":
                return TextClassificationDataModule.from_joint_config(params)
            case "listops":
                return ListopsDataModule.from_joint_config(params)
            case "cifar10":
                return CIFAR10DataModule.from_joint_config(params)
            case "pathfinder":
                return PathfinderDataModule.from_joint_config(params)
            case "text_retrieval":
                return TextRetrievalDataModule.from_joint_config(params)
            case _:
                raise ValueError(f"Unknown task: {name}")
