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
            return cls.__build_plateau(optimizer, scheduler_config, metric_to_track)
        elif scheduler_name == "cosine":
            return cls.__build_cosine(optimizer, scheduler_config)
        elif scheduler_name == "cosine_warmup":
            return cls.__build_cosine_warmup(optimizer, scheduler_config, train_steps)
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_name}")

    @classmethod
    def __build_plateau(cls, optimizer, scheduler_config, metric_to_track):
        metric, mode = metric_to_track
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_config)
        return {
            "scheduler": scheduler,
            "monitor": f"val_{metric}",
            "mode": mode,
            "interval": "epoch",
            "frequency": 1,
        }

    @classmethod
    def __build_cosine(cls, optimizer, scheduler_config):
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, **scheduler_config)
        return {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
        }

    @classmethod
    def __build_cosine_warmup(cls, optimizer, scheduler_config, train_steps):
        ratio_warmup_steps = scheduler_config.pop("ratio_warmup_steps")
        warmup_steps = int(train_steps * ratio_warmup_steps)

        linear_params = {
            "start_factor": scheduler_config.pop("start_factor"),
            "end_factor": scheduler_config.pop("end_factor"),
            "total_iters": warmup_steps,
        }

        linear_lr = lr_scheduler.LinearLR(optimizer, **linear_params)
        cosine_anneal_lr = lr_scheduler.CosineAnnealingLR(optimizer, **scheduler_config)

        scheduler = lr_scheduler.SequentialLR(
            optimizer,
            [linear_lr, cosine_anneal_lr],
            [warmup_steps],
        )

        return {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
        }
