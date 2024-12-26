import json
import warnings
from time import time
from typing import Dict, Tuple, Literal
from pathlib import Path
from datetime import datetime

import torch
from pandas.errors import DtypeWarning
from sklearn.metrics import f1_score, accuracy_score, root_mean_squared_error
from pytorch_widedeep import Trainer
from pytorch_widedeep.losses import RMSELoss
from pytorch_widedeep.models import TabMlp, WideDeep
from pytorch_widedeep.metrics import F1Score, Accuracy
from pytorch_widedeep.callbacks import LRHistory, EarlyStopping
from pytorch_widedeep.preprocessing import TabPreprocessor

from rec_tools.dl_tools import set_optim, set_scheduler
from rec_tools.constants import RESULTS_DIR
from rec_tools.dl_tools.losses import SigmoidBoundedRMSELoss
from rec_tools.dl_tools.configs.tabmlp_configs import (
    TabMlpConfig,
    get_all_config_combinations,
)
from rec_tools.prepare_experiments.prepare_ts_or_li import (
    experiment_without_feat_engineering,
)

warnings.filterwarnings("ignore", category=DtypeWarning)


def train_tabmlp(
    config: TabMlpConfig,
    split_type: Literal["ts", "li"] = "ts",
    binary_target: bool = True,
    sigmoid_bounded_rmse: bool = False,
    verbose: bool = False,
) -> Tuple[WideDeep, Dict[str, float]]:
    train_df, val_df, _cat_cols = experiment_without_feat_engineering(
        split_type, binary_target
    )

    tab_preprocessor_config = config.tab_preprocessor
    cat_embed_cols = (
        _cat_cols
        if tab_preprocessor_config.cat_embed_cols == "all"
        else tab_preprocessor_config.cat_embed_cols
    )
    tab_preprocessor = TabPreprocessor(
        cat_embed_cols=cat_embed_cols,
        embedding_rule=tab_preprocessor_config.embedding_rule,
    )

    X_train = tab_preprocessor.fit_transform(train_df)
    X_val = tab_preprocessor.transform(val_df)

    tab_mlp = TabMlp(
        column_idx=tab_preprocessor.column_idx,
        cat_embed_input=tab_preprocessor.cat_embed_input,
        continuous_cols=tab_preprocessor.continuous_cols,
        **config.mlp.model_dump(),
    )

    model = WideDeep(deeptabular=tab_mlp)

    if not binary_target and sigmoid_bounded_rmse:
        objective = "regression"
        custom_loss = SigmoidBoundedRMSELoss()
    elif not binary_target:
        objective = "regression"
        custom_loss = RMSELoss()
    else:
        objective = "binary"
        custom_loss = None

    optimizer = set_optim(model, config.optimizer.model_dump(), verbose=verbose)

    steps_per_epoch = len(train_df) // config.batch_size
    steps_per_epoch += len(train_df) % config.batch_size != 0
    lr_scheduler = set_scheduler(
        optimizer,
        steps_per_epoch,
        config.lr_scheduler.model_dump(),
        verbose=verbose,
    )
    early_stopping = EarlyStopping(**config.early_stopping.model_dump())
    lr_history = LRHistory(n_epochs=config.n_epochs)

    metrics = [Accuracy(), F1Score()] if binary_target else None

    trainer = Trainer(
        model=model,
        objective=objective,
        custom_loss=custom_loss,
        optimizers=optimizer,
        lr_schedulers=lr_scheduler,
        callbacks=[early_stopping, lr_history],
        metrics=metrics,
    )

    start_time = time()
    trainer.fit(
        X_train={"X_tab": X_train, "target": train_df["rating"]},
        X_val={"X_tab": X_val, "target": val_df["rating"]},
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
    )
    train_time = time() - start_time
    preds = trainer.predict(X_tab=X_val)

    if binary_target:
        acc = accuracy_score(val_df["rating"], preds)
        f1 = f1_score(val_df["rating"], preds)
        val_loss = early_stopping.best
        best_epoch = early_stopping.best_epoch
        print("Best Epoch: ", best_epoch + 1)
        print(f"TabMLP Accuracy: {acc:.4f}")
        print(f"TabMLP F1: {f1:.4f}")
        print(f"TabMLP Val Loss: {val_loss:.4f}")
        return model, {
            "accuracy": acc,
            "f1": f1,
            "val_loss": val_loss,
            "best_epoch": best_epoch,
            "train_time": train_time,
        }
    else:
        rmse = root_mean_squared_error(val_df["rating"], preds)
        val_loss = early_stopping.best
        best_epoch = early_stopping.best_epoch
        print("Best Epoch: ", best_epoch + 1)
        print(f"TabMLP RMSE: {rmse:.4f}")
        print(f"TabMLP Val Loss: {val_loss:.4f}")
        return model, {
            "rmse": rmse,
            "val_loss": val_loss,
            "best_epoch": best_epoch,
            "train_time": train_time,
        }


def main(
    split_type: Literal["ts", "li"] = "ts",
    binary_target: bool = True,
) -> None:
    configs = get_all_config_combinations()
    for config_name, config in configs.items():
        print("-" * 100)
        print(f"Training {config_name}...")
        print("-" * 100)

        model, metrics = train_tabmlp(config, split_type, binary_target)

        print("-" * 100)
        print(f"Training {config_name}... done")
        print("-" * 100)

    suffix = str(datetime.now()).replace(" ", "_").split(".")[:-1][0]
    results_dir = (
        Path(RESULTS_DIR)
        / f"results_tabmlp_{split_type}_{'binary' if binary_target else 'regression'}"
    )
    full_results_dir = results_dir / f"tabmlp_{suffix}"
    full_results_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), full_results_dir / "model.pth")
    with open(full_results_dir / "results.json", "w") as f:
        json.dump(metrics, f, indent=4)
    with open(full_results_dir / "config.json", "w") as f:
        json.dump(config, f, indent=4)


if __name__ == "__main__":
    main(split_type="ts", binary_target=True)
    main(split_type="li", binary_target=True)
    main(split_type="ts", binary_target=False)
    main(split_type="li", binary_target=False)
