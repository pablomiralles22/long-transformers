import numpy as np

from torch.optim import lr_scheduler


class SchedulerBuilder:
    @classmethod
    def build(
        cls,
        optimizer,
        scheduler_config: dict,
        train_steps: int = None,
    ):
        scheduler_type = scheduler_config.pop("_type_")

        if scheduler_type == "plateau":
            return cls.__build_plateau(optimizer, scheduler_config)
        elif scheduler_type == "cosine":
            return cls.__basic_build(lr_scheduler.CosineAnnealingWarmRestarts, optimizer, scheduler_config)
        elif scheduler_type == "constant":
            return cls.__total_iters_build(lr_scheduler.ConstantLR, optimizer, scheduler_config, train_steps)
        elif scheduler_type == "linear":
            return cls.__total_iters_build(lr_scheduler.LinearLR, optimizer, scheduler_config, train_steps)
        elif scheduler_type == "sequential":
            return cls.__build_sequential(optimizer, scheduler_config, train_steps)
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_type}")

    @classmethod
    def __build_plateau(cls, optimizer, scheduler_config):
        scheduler_params = scheduler_config.get("scheduler_params", {})

        monitor = scheduler_params.pop("metric", "train_loss")
        mode = scheduler_params.get("mode", "min")

        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_params)
        return {
            "scheduler": scheduler,
            "monitor": monitor,
            "mode": mode,
            "interval": scheduler_config.get("interval", "epoch"),
            "frequency": scheduler_config.get("frequency", 1),
        }

    @classmethod
    def __basic_build(cls, scheduler_cls, optimizer, scheduler_config):
        scheduler_params = scheduler_config.get("scheduler_params", {})
        scheduler = scheduler_cls(optimizer, **scheduler_params)
        return {
            "scheduler": scheduler,
            "interval": scheduler_config.get("interval", "step"),
            "frequency": scheduler_config.get("frequency", 1),
        }

    @classmethod
    def __total_iters_build(cls, scheduler_cls, optimizer, scheduler_config, train_steps):
        scheduler_params = scheduler_config.get("scheduler_params", {})
        scheduler_params["total_iters"] = train_steps
        scheduler = scheduler_cls(optimizer, **scheduler_params)
        return {
            "scheduler": scheduler,
            "interval": scheduler_config.get("interval", "step"),
            "frequency": scheduler_config.get("frequency", 1),
        }

    @classmethod
    def __build_sequential(cls, optimizer, scheduler_config, train_steps):
        scheduler_params = scheduler_config.get("scheduler_params", {})
        subschedulers_params = scheduler_params["schedulers"]

        train_steps_per_schedulers = [
            int(train_steps * subscheduler_params.pop("step_ratio"))
            for subscheduler_params in subschedulers_params
        ]
        train_steps_per_schedulers[-1] = train_steps - sum(train_steps_per_schedulers[:-1])

        milestones = np.cumsum(train_steps_per_schedulers).tolist()[:-1]
        schedulers = []

        for subscheduler_params, num_steps in zip(subschedulers_params, train_steps_per_schedulers):
            subscheduler_params["interval"] = scheduler_params.get("interval", "step")
            subscheduler_params["frequency"] = scheduler_params.get("frequency", 1)

            scheduler_dict = cls.build(optimizer, subscheduler_params, num_steps)
            schedulers.append(scheduler_dict["scheduler"])
        
        sequential_scheduler = lr_scheduler.SequentialLR(optimizer, schedulers, milestones)

        return {
            "scheduler": sequential_scheduler,
            "interval": scheduler_params.get("interval", "step"),
            "frequency": scheduler_params.get("frequency", 1),
        }