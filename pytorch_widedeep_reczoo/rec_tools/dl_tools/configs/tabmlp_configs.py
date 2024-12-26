from typing import Dict, List, Union, Literal, Optional
from itertools import product

from pydantic import BaseModel

from rec_tools.dl_tools.configs.base_configs import (
    N_EPOCHS,
    BATCH_SIZE,
    CyclicLRParams,
    OptimizerConfig,
    SchedulerConfig,
    OneCycleLRParams,
    EarlyStoppingConfig,
    CosineAnnealingLRParams,
    ReduceLROnPlateauParams,
    CosineAnnealingWarmRestartsParams,
)

CAT_EMBED_COLS_OPTIONS: List[Union[Literal["all"], List[str]]] = [
    "all",
    ["user_id", "item_id"],
]
EMBEDDING_RULE_OPTIONS: List[Literal["fastai_new", "fastai_old"]] = [
    "fastai_new",
    "fastai_old",
]
ACTIVATION_OPTIONS: List[Literal["relu", "leakyrelu"]] = ["relu", "leakyrelu"]
MLP_HIDDEN_DIMS_AND_DROPOUT_OPTIONS = [
    ([50, 50], 0.1),
    ([100, 100], 0.1),
    ([200, 200], 0.2),
    ([100, 400, 100], 0.2),
]
SCHEDULER_OPTIONS: List[
    Literal["reduce_on_plateau", "onecycle", "cyclic", "cosine", "cosine_warm_restarts"]
] = [
    "reduce_on_plateau",
    "onecycle",
    "cyclic",
    "cosine",
    "cosine_warm_restarts",
]

# as it is we have a total of 2 * 2 * 2 * 4 * 5 = 160 configurations


class MlpConfig(BaseModel):
    mlp_hidden_dims: List[int] = [200, 200]
    mlp_activation: Literal["relu", "leakyrelu"] = "relu"
    mlp_dropout: float = 0.1


class TabPreprocessorConfig(BaseModel):
    cat_embed_cols: Literal["all"] | List[str] = "all"
    embedding_rule: Literal["fastai_new", "fastai_old"] = "fastai_new"


class TabMlpConfig(BaseModel):
    tab_preprocessor: TabPreprocessorConfig
    mlp: MlpConfig
    optimizer: OptimizerConfig
    lr_scheduler: SchedulerConfig
    early_stopping: EarlyStoppingConfig
    batch_size: int = BATCH_SIZE
    n_epochs: int = N_EPOCHS


def get_all_config_combinations(
    manual_configs: Optional[Dict[str, TabMlpConfig]] = None
) -> Dict[str, TabMlpConfig]:
    configs = {}

    combinations = product(
        CAT_EMBED_COLS_OPTIONS,
        EMBEDDING_RULE_OPTIONS,
        MLP_HIDDEN_DIMS_AND_DROPOUT_OPTIONS,
        ACTIVATION_OPTIONS,
        SCHEDULER_OPTIONS,
    )

    for idx, (
        cat_cols,
        emb_rule,
        (hidden_dims, dropout),
        activation,
        scheduler,
    ) in enumerate(combinations):
        config_name = f"tabmlp_config_{idx + 1}"

        # I could be a bit more flexible and type Dict[str, BaseModel]
        scheduler_params: Dict[
            str,
            ReduceLROnPlateauParams
            | OneCycleLRParams
            | CyclicLRParams
            | CosineAnnealingLRParams
            | CosineAnnealingWarmRestartsParams,
        ] = {
            "reduce_on_plateau": ReduceLROnPlateauParams(),
            "onecycle": OneCycleLRParams(),
            "cyclic": CyclicLRParams(),
            "cosine": CosineAnnealingLRParams(),
            "cosine_warm_restarts": CosineAnnealingWarmRestartsParams(),
        }

        configs[config_name] = TabMlpConfig(
            tab_preprocessor=TabPreprocessorConfig(
                cat_embed_cols=cat_cols,
                embedding_rule=emb_rule,
            ),
            mlp=MlpConfig(
                mlp_hidden_dims=hidden_dims,
                mlp_activation=activation,
                mlp_dropout=dropout,
            ),
            optimizer=OptimizerConfig(),
            lr_scheduler=SchedulerConfig(
                type=scheduler,
                params=scheduler_params[scheduler],
            ),
            early_stopping=EarlyStoppingConfig(),
            batch_size=BATCH_SIZE,
            n_epochs=N_EPOCHS,
        )

    if manual_configs is not None:
        configs.update(manual_configs)

    return configs


# Calling the function just for testing. In the main file we import the function
CONFIGURATIONS = get_all_config_combinations()
