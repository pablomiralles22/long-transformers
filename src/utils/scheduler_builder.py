import numpy as np

from torch.optim import lr_scheduler


class SchedulerBuilder:
    @classmethod
    def build(
        cls,
        optimizer,
        scheduler_config: dict,
        metric_to_track: tuple[str, str],
        train_steps: int = None,
    ):
        scheduler_name = scheduler_config.pop("_name_")

        if scheduler_name == "plateau":
            return cls.__build_plateau(optimizer, scheduler_config)
        elif scheduler_name == "cosine":
            return cls.__build_cosine(optimizer, scheduler_config)
        elif scheduler_name == "linear":
            return cls.__build_linear(optimizer, scheduler_config, train_steps)
        elif scheduler_name == "sequential":
            return cls.__build_sequential(optimizer, scheduler_config, metric_to_track, train_steps)
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_name}")

    @classmethod
    def __build_plateau(cls, optimizer, scheduler_config):
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_config)
        return {
            "scheduler": scheduler,
            "monitor": "train_loss",
            "mode": "min",
            "interval": "epoch",
            "frequency": 1,
        }

    @classmethod
    def __build_cosine(cls, optimizer, scheduler_config):
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, **scheduler_config)
        return {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
        }

    @classmethod
    def __build_linear(cls, optimizer, scheduler_config, train_steps):
        scheduler_config["total_iters"] = train_steps
        scheduler = lr_scheduler.LinearLR(optimizer, **scheduler_config)
        return {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
        }

    @classmethod
    def __build_sequential(cls, optimizer, scheduler_config, metric_to_track, train_steps):
        schedulers_params = scheduler_config["schedulers"].values()

        train_steps_per_schedulers = [
            int(train_steps * scheduler_params.pop("step_ratio"))
            for scheduler_params in schedulers_params
        ]
        train_steps_per_schedulers[-1] = train_steps - sum(train_steps_per_schedulers[:-1])

        milestones = np.cumsum(train_steps_per_schedulers).tolist()[:-1]
        schedulers = []
        intervals = set()

        for scheduler, num_steps in zip(schedulers_params, train_steps_per_schedulers):
            scheduler_dict = cls.build(optimizer, scheduler, metric_to_track, num_steps)
            intervals.add(scheduler_dict["interval"])
            schedulers.append(scheduler_dict["scheduler"])
        
        assert len(intervals) == 1, "All schedulers must have the same interval"

        sequential_scheduler = lr_scheduler.SequentialLR(optimizer, schedulers, milestones)

        return {
            "scheduler": sequential_scheduler,
            "interval": intervals.pop(),
            "frequency": 1,
        }