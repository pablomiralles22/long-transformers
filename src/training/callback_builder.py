from pytorch_lightning.callbacks import (
    Callback,
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    StochasticWeightAveraging,
)


class CallbackBuilder:
    @staticmethod
    def build_callback(
        callback_params: list[dict], metric_to_track: str, mode: str
    ) -> list[Callback]:
        name = callback_params.pop("_name_")
        if name == "model_checkpoint":
            filename = "{epoch}-{METRIC:.2f}".replace("METRIC", metric_to_track)
            return ModelCheckpoint(
                **callback_params,
                monitor=metric_to_track,
                mode=mode,
                filename=filename,
            )
        elif name == "early_stopping":
            return EarlyStopping(**callback_params, monitor=metric_to_track, mode=mode)
        elif name == "learning_rate_monitor":
            return LearningRateMonitor(**callback_params)
        elif name == "stochastic_weight_averaging":
            return StochasticWeightAveraging(**callback_params)
        else:
            raise ValueError(f"Unknown callback: {name}")
