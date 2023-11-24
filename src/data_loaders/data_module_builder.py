from src.data_loaders.text_classification import TextClassificationDataModule


class DataModuleBuilder:
    @classmethod
    def build_data_module(cls, name, params):
        match name:
            case "text-classification":
                return TextClassificationDataModule.from_joint_config(params)
            case _:
                raise ValueError(f"Unknown task: {name}")
