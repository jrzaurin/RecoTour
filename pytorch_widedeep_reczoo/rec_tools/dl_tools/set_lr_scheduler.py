import warnings
from typing import Any, Dict, Tuple, Union

import torch
from torch.optim.lr_scheduler import (
    CyclicLR,
    OneCycleLR,
    CosineAnnealingLR,
    ReduceLROnPlateau,
    CosineAnnealingWarmRestarts,
)

SchedulerType = Union[
    ReduceLROnPlateau,
    CyclicLR,
    OneCycleLR,
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
]


def set_scheduler(
    optimizer: torch.optim.Optimizer,
    steps_per_epoch: int,
    lr_scheduler_config: Dict[str, Any] | None,
    verbose: bool = True,
    n_epochs: int | None = None,
) -> SchedulerType | None:
    if lr_scheduler_config is None:
        if verbose:
            warnings.warn(
                "No lr scheduler found for this experiment. The default ReduceLROnPlateau scheduler will be used."
            )
        return None
    else:
        scheduler_type = lr_scheduler_config.get("type").lower()
        scheduler_params = lr_scheduler_config.get("params")

        if scheduler_type == "reduce_on_plateau":
            scheduler: SchedulerType = ReduceLROnPlateau(
                optimizer,
                **scheduler_params,
            )
        elif scheduler_type == "cyclic":
            assert n_epochs is not None
            step_size_up, step_size_down = _steps_up_down(
                steps_per_epoch,
                n_epochs,
                scheduler_params.pop("pct_step_up"),
                scheduler_params.pop("n_cycles"),
            )
            scheduler = CyclicLR(
                optimizer,
                step_size_up=step_size_up,
                step_size_down=step_size_down,
                **scheduler_params,
            )
        elif scheduler_type == "onecycle":
            assert n_epochs is not None
            total_steps = steps_per_epoch * n_epochs
            scheduler = OneCycleLR(
                optimizer,
                total_steps=total_steps,
                **scheduler_params,
            )
        elif scheduler_type == "cosine":
            scheduler = CosineAnnealingLR(
                optimizer,
                **scheduler_params,
            )
        elif scheduler_type == "cosine_warm_restarts":
            scheduler = CosineAnnealingWarmRestarts(
                optimizer,
                **scheduler_params,
            )
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")

        return scheduler


def _steps_up_down(
    steps_per_epoch: int, n_epochs: int, pct_step_up: float, n_cycles: int
) -> Tuple[int, int]:
    total_steps = steps_per_epoch * n_epochs
    steps_per_cycle = total_steps // n_cycles
    step_size_up = round(steps_per_cycle * pct_step_up)
    step_size_down = int(steps_per_cycle - step_size_up)
    return step_size_up, step_size_down
