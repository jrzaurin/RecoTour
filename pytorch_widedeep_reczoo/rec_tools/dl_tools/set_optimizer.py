import warnings
from typing import Any, Dict

import torch
from torch.optim import Optimizer


def set_optim(
    model, optimizer_config: Dict[str, Any] | None, verbose: bool = True
) -> Optimizer | None:
    if optimizer_config is None:
        if verbose:
            warnings.warn(
                "No optimizer found for this experiment. The default Adam optimizer will be used."
            )
        return None
    else:
        optimizer_type = optimizer_config.get("type").lower()
        optimizer_params = optimizer_config.get("params")

        if optimizer_type == "adam":
            return torch.optim.Adam(
                model.parameters(),
                **optimizer_params,
            )
        elif optimizer_type == "adamw":
            return torch.optim.AdamW(
                model.parameters(),
                **optimizer_params,
            )
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
