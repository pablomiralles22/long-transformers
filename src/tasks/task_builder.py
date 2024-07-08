from src.tasks.task import Task
from src.tasks.classification_task import ClassificationTask
from src.tasks.denoise_task import DenoiseTask

class TaskBuilder:
    @staticmethod
    def build_task(task_params: dict) -> Task:
        task_name = task_params.pop("_name_")
        if task_name == "classification":
            return ClassificationTask(**task_params)
        elif task_name == "denoise":
            return DenoiseTask(**task_params)
        else:
            raise ValueError(f"Task {task_name} not supported")
