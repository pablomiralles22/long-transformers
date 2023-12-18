from src.data_loaders.text_classification import TextClassificationDataModule
from src.data_loaders.listops import ListopsDataModule


class DataModuleBuilder:
    @classmethod
    def build_data_module(cls, name: str, params: dict):
        match name:
            case "text-classification":
                return TextClassificationDataModule.from_joint_config(params)
            case "listops":
                return ListopsDataModule.from_joint_config(params)
            case _:
                raise ValueError(f"Unknown task: {name}")
