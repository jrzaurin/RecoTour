import os
import pickle
from typing import Any, Dict, List, Optional
from pathlib import Path

import ray
import fire
import numpy as np
import mlflow
import pandas as pd
import lightgbm as lgb
from ray import tune, train
from sklearn.metrics import f1_score, accuracy_score
from ray.tune.schedulers import HyperBandScheduler
from ray.tune.search.hyperopt import HyperOptSearch

n_jobs = os.cpu_count()


def load_data(
    split_dir: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str]]:
    """
    Load train and validation data from specified directory and prepare for LightGBM

    Args:
        split_dir: Directory name suffix for train_val_test_splits_{split_dir}

    Returns:
        X_train, X_val: Features for training and validation
        y_train, y_val: Target variables
        cat_cols: List of categorical column names
    """
    # Load data
    base_path = Path(f"train_val_test_splits_{split_dir}")
    train_df = pd.read_csv(base_path / "train.csv")
    val_df = pd.read_csv(base_path / "val.csv")

    # Load feature engineering artifacts
    with open("artifacts/feature_engineer.pkl", "rb") as f:
        fe_artifact = pickle.load(f)
    cat_cols = fe_artifact.cat_cols

    # Separate features and target
    y_train = train_df["rating"]
    y_val = val_df["rating"]
    X_train = train_df.drop("rating", axis=1)
    X_val = val_df.drop("rating", axis=1)

    return X_train, X_val, y_train, y_val, cat_cols


def train_lgbm(
    config: Dict[str, Any],
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    cat_cols: List[str],
    track_with_mlflow: bool,
) -> None:
    """
    Train LightGBM model with given configuration
    Also serves as the objective function for Ray Tune
    """
    train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_cols)
    val_data = lgb.Dataset(X_val, label=y_val, categorical_feature=cat_cols)

    params = {
        "objective": "multiclass",
        "num_class": len(np.unique(y_train)),
        "metric": "multi_logloss",
        "verbose": -1,
        **config,
    }

    model = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    y_pred = model.predict(X_val)
    y_pred_labels = np.argmax(y_pred, axis=1)

    accuracy = accuracy_score(y_val, y_pred_labels)
    f1 = f1_score(y_val, y_pred_labels, average="weighted")

    if track_with_mlflow:
        mlflow.log_params(config)
        mlflow.log_metrics({"accuracy": accuracy, "f1_score": f1})

    train.report({"accuracy": accuracy, "f1_score": f1})


def run_optimization(
    data_path: str,
    optimizer: str = "tpe",
    num_trials: int = 100,
    track_with_mlflow: bool = False,
    experiment_name: Optional[str] = None,
    mlflow_tracking_uri: Optional[str] = None,
) -> None:
    """
    Main function to run hyperparameter optimization

    Args:
        data_path: Path to the dataset
        optimizer: Optimization algorithm ('tpe' or 'hyperband')
        num_trials: Number of trials for optimization
        track_with_mlflow: Whether to track experiments with MLflow
        experiment_name: Name of the MLflow experiment
        mlflow_tracking_uri: MLflow tracking URI
    """

    # MLflow setup
    if track_with_mlflow:
        if mlflow_tracking_uri:
            mlflow.set_tracking_uri(mlflow_tracking_uri)
        if experiment_name:
            mlflow.set_experiment(experiment_name)

    # Load and split data
    X_train, X_val, y_train, y_val, cat_cols = load_data(data_path)

    # Define search space
    search_space = {
        "learning_rate": tune.loguniform(1e-4, 1e-1),
        "num_leaves": tune.randint(20, 200),
        "max_depth": tune.randint(3, 12),
        "min_data_in_leaf": tune.randint(5, 100),
        "feature_fraction": tune.uniform(0.5, 1.0),
        "bagging_fraction": tune.uniform(0.5, 1.0),
        "bagging_freq": tune.randint(1, 7),
        "min_child_samples": tune.randint(5, 100),
    }

    # Initialize Ray
    ray.init()

    # Set up optimizer
    if optimizer.lower() == "tpe":
        search_alg = HyperOptSearch(metric="accuracy", mode="max")
        scheduler = None
    else:  # hyperband
        search_alg = HyperOptSearch(metric="accuracy", mode="max")
        scheduler = HyperBandScheduler(metric="accuracy", mode="max")

    # Run optimization
    analysis = tune.run(
        tune.with_parameters(
            train_lgbm,
            X_train=X_train,
            X_val=X_val,
            y_train=y_train,
            y_val=y_val,
            track_with_mlflow=track_with_mlflow,
        ),
        config=search_space,
        search_alg=search_alg,
        scheduler=scheduler,
        num_samples=num_trials,
        resources_per_trial={"cpu": n_jobs},
        local_dir="./ray_results",
        name="lightgbm_optimization",
    )

    # Get best results
    best_trial = analysis.best_trial
    print(f"\nBest trial config: {best_trial.config}")
    print(f"Best trial final validation accuracy: {best_trial.last_result['accuracy']}")
    print(f"Best trial final validation f1 score: {best_trial.last_result['f1_score']}")

    # Shutdown Ray
    ray.shutdown()


def main(
    data_path: str,
    optimizer: str = "tpe",
    num_trials: int = 50,
    track_with_mlflow: bool = False,
    experiment_name: Optional[str] = None,
    mlflow_tracking_uri: Optional[str] = None,
) -> None:
    """
    Main entry point using Google Fire

    Args:
        data_path: Path to the dataset
        optimizer: Optimization algorithm ('tpe' or 'hyperband')
        num_trials: Number of trials for optimization
        track_with_mlflow: Whether to track experiments with MLflow
        experiment_name: Name of the MLflow experiment
        mlflow_tracking_uri: MLflow tracking URI
    """
    run_optimization(
        data_path=data_path,
        optimizer=optimizer,
        num_trials=num_trials,
        track_with_mlflow=track_with_mlflow,
        experiment_name=experiment_name,
        mlflow_tracking_uri=mlflow_tracking_uri,
    )


if __name__ == "__main__":
    fire.Fire(main)
