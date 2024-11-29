import os
import json
import warnings
from typing import Any, Dict, List, Literal, Optional
from pathlib import Path

import ray
import mlflow
import pandas as pd
import lightgbm as lgb
from ray import tune, train
from sklearn.metrics import f1_score, accuracy_score
from ray.tune.schedulers import HyperBandScheduler
from pytorch_widedeep.utils import LabelEncoder
from ray.tune.search.hyperopt import HyperOptSearch

from rec_tools.constants import RESULTS_DIR
from rec_tools.prepare_experiments.prepare_ts_or_li import (
    experiment_with_feature_engineering,
)

warnings.filterwarnings("ignore")


def cleanup_old_trials(best_trial_path: str) -> None:
    for trial_dir in Path(best_trial_path).parent.glob("train_*"):
        if trial_dir != Path(best_trial_path):
            import shutil

            shutil.rmtree(trial_dir)


def train_lgbm(
    config: Dict[str, Any],
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    cat_cols: List[str],
    track_with_mlflow: bool,
) -> None:

    train_data = lgb.Dataset(
        X_train, label=y_train, categorical_feature=cat_cols, free_raw_data=False
    )
    val_data = lgb.Dataset(
        X_val, label=y_val, reference=train_data, free_raw_data=False
    )

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbose": -1,
        **config,
    }

    model = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50)],
    )

    y_pred = model.predict(X_val)
    y_pred_labels = (y_pred > 0.5).astype(int)  # type: ignore
    valid_loss = model.best_score["valid_0"]["binary_logloss"]

    accuracy = accuracy_score(y_val, y_pred_labels)
    f1 = f1_score(y_val, y_pred_labels)

    if track_with_mlflow:
        mlflow.log_params(config)
        mlflow.log_metrics({"accuracy": accuracy, "f1_score": f1})

    train.report({"accuracy": accuracy, "f1_score": f1, "val_loss": valid_loss})


def run_optimization(
    split_type: Literal["ts", "li"] = "ts",
    use_umap: Literal["st", "ch"] = "ch",
    optimizer: Literal["tpe", "hyperband"] = "tpe",
    num_trials: int = 200,
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

    train_df, val_df, cat_cols = experiment_with_feature_engineering(
        use_umap, split_type
    )

    encoder = LabelEncoder(columns_to_encode=cat_cols)

    train_df_encoded = encoder.fit_transform(train_df)
    val_df_encoded = encoder.transform(val_df)

    y_train = train_df_encoded["rating"]
    y_val = val_df_encoded["rating"]
    X_train = train_df_encoded.drop("rating", axis=1)
    X_val = val_df_encoded.drop("rating", axis=1)

    # Define search space
    search_space = {
        "learning_rate": tune.loguniform(1e-4, 3e-1),
        "num_iterations": tune.qrandint(100, 1000, 50),
        "num_leaves": tune.qrandint(20, 200, 10),
        "min_data_in_leaf": tune.qrandint(5, 100, 10),
        "feature_fraction": tune.uniform(0.4, 1.0),
        "min_child_samples": tune.qrandint(5, 100, 10),
        "lambda_l1": tune.loguniform(1e-4, 1.0),
        "lambda_l2": tune.loguniform(1e-4, 1.0),
    }

    try:
        ray.init(ignore_reinit_error=True)

        if optimizer.lower() == "tpe":
            search_alg = HyperOptSearch(metric="val_loss", mode="min")
            scheduler = None
        else:  # hyperband
            search_alg = HyperOptSearch(metric="val_loss", mode="min")
            scheduler = HyperBandScheduler(metric="val_loss", mode="min")

        results_dir = os.path.abspath(
            "/".join([RESULTS_DIR, f"results_lgb_with_ray_{use_umap}_{split_type}"])
        )

        tuner = tune.Tuner(
            tune.with_parameters(
                train_lgbm,
                X_train=X_train,
                X_val=X_val,
                y_train=y_train,
                y_val=y_val,
                cat_cols=encoder.columns_to_encode,
                track_with_mlflow=track_with_mlflow,
            ),
            tune_config=tune.TuneConfig(
                metric="val_loss" if optimizer.lower() == "tpe" else None,
                mode="min" if optimizer.lower() == "tpe" else None,
                search_alg=search_alg,
                scheduler=scheduler,
                num_samples=num_trials,
            ),
            param_space=search_space,
            run_config=train.RunConfig(
                storage_path=results_dir,
                name=f"results_lgbm_ray_{optimizer}",
                verbose=0,
            ),
        )

        results = tuner.fit()
        best_result = results.get_best_result(
            metric="val_loss", mode="min", scope="all"
        )

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
            Path(results_dir)
            / f"results_lgbm_ray_{optimizer}"
            / "best_experiment_info.json",
            "w",
        ) as f:
            json.dump(best_experiment_info, f, indent=4)

        cleanup_old_trials(best_result.path)

    finally:
        ray.shutdown()


if __name__ == "__main__":

    # run_optimization(optimizer="tpe", split_type="ts")
    # run_optimization(optimizer="tpe", split_type="li")
    run_optimization(optimizer="hyperband", split_type="ts")
    run_optimization(optimizer="hyperband", split_type="li")
