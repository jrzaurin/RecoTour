import os
import warnings
from typing import Any, Dict, List, Literal, Optional

import ray
import mlflow
import pandas as pd
from ray import tune, train
from catboost import Pool, CatBoostClassifier
from sklearn.metrics import f1_score, accuracy_score
from ray.tune.schedulers import HyperBandScheduler
from ray.tune.search.hyperopt import HyperOptSearch

from rec_zoo.prepare_experiments.prepare_ts import prepare_experiment

warnings.filterwarnings("ignore")


def train_catboost(
    config: Dict[str, Any],
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    cat_cols: List[str],
    track_with_mlflow: bool,
) -> None:

    train_pool = Pool(X_train, label=y_train, cat_features=cat_cols)
    val_pool = Pool(X_val, label=y_val, cat_features=cat_cols)

    model = CatBoostClassifier(
        iterations=config["iterations"],
        learning_rate=config["learning_rate"],
        depth=config["depth"],
        l2_leaf_reg=config["l2_leaf_reg"],
        verbose=0,
    )

    model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=100)

    y_pred = model.predict(X_val)
    y_pred_labels = y_pred.astype(int)
    valid_loss = model.get_best_score()["validation"]["Logloss"]

    accuracy = accuracy_score(y_val, y_pred_labels)
    f1 = f1_score(y_val, y_pred_labels)

    if track_with_mlflow:
        mlflow.log_params(config)
        mlflow.log_metrics({"accuracy": accuracy, "f1_score": f1})

    train.report({"accuracy": accuracy, "f1_score": f1, "val_loss": valid_loss})


def run_optimization(
    optimizer: Literal["tpe", "hyperband"] = "tpe",
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

    train_df, val_df, _, encoder = prepare_experiment(use_umap="ch", gbm="catboost")
    y_train = train_df["rating"]
    y_val = val_df["rating"]
    X_train = train_df.drop("rating", axis=1)
    X_val = val_df.drop("rating", axis=1)

    # Define search space
    search_space = {
        "learning_rate": tune.loguniform(1e-4, 3e-1),
        "iterations": tune.qrandint(100, 1000, 50),
        "depth": tune.randint(4, 10),
        "l2_leaf_reg": tune.loguniform(1e-4, 1e-1),
        "feature_fraction": tune.uniform(0.5, 1.0),
        "min_data_in_leaf": tune.qrandint(5, 50, 5),
    }

    try:
        ray.init(ignore_reinit_error=True)

        if optimizer.lower() == "tpe":
            search_alg = HyperOptSearch(metric="accuracy", mode="max")
            scheduler = None
        else:  # hyperband
            search_alg = HyperOptSearch(metric="accuracy", mode="max")
            scheduler = HyperBandScheduler(metric="accuracy", mode="max")

        tuner = tune.Tuner(
            tune.with_parameters(
                train_catboost,
                X_train=X_train,
                X_val=X_val,
                y_train=y_train,
                y_val=y_val,
                cat_cols=encoder.columns_to_encode,
                track_with_mlflow=track_with_mlflow,
            ),
            tune_config=tune.TuneConfig(
                metric="accuracy",
                mode="max",
                search_alg=search_alg,
                scheduler=scheduler,
                num_samples=num_trials,
            ),
            param_space=search_space,
            run_config=ray.air.RunConfig(
                storage_path=os.path.abspath("./results"),
                name=f"results_catboost_ray_{optimizer}",
                verbose=0,
                checkpoint_config=ray.air.CheckpointConfig(num_to_keep=1),
            ),
        )

        results = tuner.fit()
        best_result = results.get_best_result()

        print(f"\nBest trial config: {best_result.config}")
        print(
            f"Best trial final validation accuracy: {best_result.metrics['accuracy']}"
        )
        print(
            f"Best trial final validation f1 score: {best_result.metrics['f1_score']}"
        )

    finally:
        ray.shutdown()


if __name__ == "__main__":

    run_optimization()
