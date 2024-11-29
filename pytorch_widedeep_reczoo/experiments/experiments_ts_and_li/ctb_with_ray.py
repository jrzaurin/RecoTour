import os
import json
import warnings
from typing import Any, Dict, List, Literal, Optional
from pathlib import Path

import ray
import mlflow
import pandas as pd
import catboost as ctb
from ray import tune, train
from sklearn.metrics import f1_score, accuracy_score
from ray.tune.schedulers import HyperBandScheduler
from ray.tune.search.hyperopt import HyperOptSearch

from rec_tools.constants import RESULTS_DIR
from rec_tools.prepare_experiments.prepare_ts_or_li import (
    experiment_with_feat_engineering,
)

warnings.filterwarnings("ignore")


def cleanup_old_trials(best_trial_path: str) -> None:
    for trial_dir in Path(best_trial_path).parent.glob("train_*"):
        if trial_dir != Path(best_trial_path):
            import shutil

            shutil.rmtree(trial_dir)


def train_catboost(
    config: Dict[str, Any],
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    cat_cols: List[str],
    track_with_mlflow: bool,
) -> None:

    train_pool = ctb.Pool(
        X_train,
        label=y_train,
        cat_features=cat_cols,
    )
    val_pool = ctb.Pool(
        X_val,
        label=y_val,
        cat_features=cat_cols,
    )

    model = ctb.train(
        pool=train_pool,
        params={
            "iterations": 500,
            "early_stopping_rounds": 50,
            "verbose": 0,
            "loss_function": "Logloss",
            "learning_rate": config["learning_rate"],
            "depth": config["depth"],
            "l2_leaf_reg": config["l2_leaf_reg"],
            "min_data_in_leaf": config["min_data_in_leaf"],
        },
        eval_set=val_pool,
    )

    y_pred = model.predict(val_pool, prediction_type="Probability")[:, 1]
    y_pred_labels = (y_pred > 0.5).astype(int)
    valid_loss = model.get_best_score()["validation"]["Logloss"]

    accuracy = accuracy_score(y_val, y_pred_labels)
    f1 = f1_score(y_val, y_pred_labels)

    if track_with_mlflow:
        mlflow.log_params(config)
        mlflow.log_metrics({"accuracy": accuracy, "f1_score": f1})

    train.report({"accuracy": accuracy, "f1_score": f1, "val_loss": valid_loss})


def run_optimization(
    split_type: Literal["ts", "li"],
    use_umap: Literal["st", "ch"],
    optimizer: Literal["tpe", "hyperband"],
    num_trials: int = 100,
    track_with_mlflow: bool = False,
    experiment_name: Optional[str] = None,
    mlflow_tracking_uri: Optional[str] = None,
) -> None:

    # MLflow setup
    if track_with_mlflow:
        if mlflow_tracking_uri:
            mlflow.set_tracking_uri(mlflow_tracking_uri)
        if experiment_name:
            mlflow.set_experiment(experiment_name)

    train_df, val_df, cat_cols = experiment_with_feat_engineering(use_umap, split_type)

    y_train = train_df["rating"]
    y_val = val_df["rating"]
    X_train = train_df.drop("rating", axis=1)
    X_val = val_df.drop("rating", axis=1)

    # Define search space
    search_space = {
        "learning_rate": tune.loguniform(1e-3, 3e-1),
        "depth": tune.randint(4, 10),
        "l2_leaf_reg": tune.loguniform(1e-4, 1.0),
        "min_data_in_leaf": tune.qrandint(5, 50, 5),
    }

    try:
        ray.init(ignore_reinit_error=True)

        if optimizer.lower() == "tpe":
            search_alg = HyperOptSearch(metric="val_loss", mode="min")
            scheduler = None
        else:  # hyperband
            search_alg = HyperOptSearch(metric="val_loss", mode="min")
            scheduler = HyperBandScheduler(metric="val_loss", mode="min")

        # Update results directory path
        results_dir = os.path.abspath(
            "/".join([RESULTS_DIR, f"results_ctb_with_ray_{use_umap}_{split_type}"])
        )

        tuner = tune.Tuner(
            tune.with_parameters(
                train_catboost,
                X_train=X_train,
                X_val=X_val,
                y_train=y_train,
                y_val=y_val,
                cat_cols=cat_cols,
                track_with_mlflow=track_with_mlflow,
            ),
            tune_config=tune.TuneConfig(
                metric="val_loss" if optimizer == "tpe" else None,
                mode="min" if optimizer == "tpe" else None,
                search_alg=search_alg,
                scheduler=scheduler,
                num_samples=num_trials,
            ),
            param_space=search_space,
            run_config=train.RunConfig(
                storage_path=results_dir,
                name=f"results_ctb_ray_{optimizer}",
                verbose=1,
            ),
        )

        results = tuner.fit()
        best_result = results.get_best_result(
            metric="val_loss", mode="min", scope="all"
        )

        # Add best experiment info saving
        best_experiment_info = {
            "metrics": {
                "accuracy": best_result.metrics["accuracy"],
                "f1": best_result.metrics["f1_score"],
                "val_loss": best_result.metrics["val_loss"],
            },
            "config": best_result.config,
            "experiment_path": best_result.path,
        }

        with open(
            Path(results_dir) / f"results_ctb_ray_{optimizer}" / "results.json",
            "w",
        ) as f:
            json.dump(best_experiment_info, f, indent=4)

        cleanup_old_trials(best_result.path)

    finally:
        ray.shutdown()


if __name__ == "__main__":
    # run_optimization(optimizer="tpe", use_umap="ch", split_type="ts")
    # run_optimization(optimizer="tpe", use_umap="ch", split_type="li")
    run_optimization(optimizer="hyperband", use_umap="ch", split_type="ts")
    run_optimization(optimizer="hyperband", use_umap="ch", split_type="li")
