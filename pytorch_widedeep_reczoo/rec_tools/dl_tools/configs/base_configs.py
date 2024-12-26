from typing import Literal, Optional

from pydantic import BaseModel

BATCH_SIZE = 1024
N_EPOCHS = 200


class OptimizerParams(BaseModel):
    lr: float = 1e-3
    weight_decay: float = 0.0


class OptimizerConfig(BaseModel):
    type: Literal["adam", "adamw"] = "adamw"
    params: OptimizerParams = OptimizerParams()


class CyclicLRParams(BaseModel):
    base_lr: float = 1e-4
    max_lr: float = 0.01
    mode: str = "triangular2"
    n_cycles: int = 4
    pct_step_up: float = 0.5


class OneCycleLRParams(BaseModel):
    max_lr: float = 0.01
    div_factor: float = 100.0


class CosineAnnealingLRParams(BaseModel):
    T_max: int = N_EPOCHS
    eta_min: float = 1e-5


class CosineAnnealingWarmRestartsParams(BaseModel):
    T_0: int = 50
    eta_min: float = 1e-5


class ReduceLROnPlateauParams(BaseModel):
    mode: str = "min"
    factor: float = 0.2
    patience: int = (
        5  # aggressive, but for the movielens dataset we don't need to be patient
    )
    threshold_mode: str = "abs"


class SchedulerConfig(BaseModel):
    type: Literal[
        "onecycle", "cyclic", "cosine", "cosine_warm_restarts", "reduce_on_plateau"
    ]
    params: Optional[
        OneCycleLRParams
        | CyclicLRParams
        | CosineAnnealingLRParams
        | CosineAnnealingWarmRestartsParams
        | ReduceLROnPlateauParams
    ] = None


class EarlyStoppingConfig(BaseModel):
    monitor: str = "val_loss"
    min_delta: float = 0.0
    patience: int = 25
    restore_best_weights: bool = True
