import os
import pickle
from typing import Any, Dict, List, Optional
from pathlib import Path

import ray
import fire
import mlflow
import pandas as pd
from ray import tune, train
from catboost import Pool, CatBoostClassifier
from sklearn.metrics import f1_score, accuracy_score
from ray.tune.schedulers import HyperBandScheduler
from ray.tune.search.hyperopt import HyperOptSearch

n_jobs = os.cpu_count()


def load_data(
    split_dir: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str]]:
    """
    Load train and validation data from specified directory and prepare for CatBoost

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

    # Get categorical feature indices for CatBoost
    cat_features_indices = [X_train.columns.get_loc(col) for col in cat_cols]

    return X_train, X_val, y_train, y_val, cat_features_indices


def train_catboost(
    config: Dict[str, Any],
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    cat_features: List[int],
    track_with_mlflow: bool,
) -> None:
    """
    Train CatBoost model with given configuration
    Also serves as the objective function for Ray Tune
    """
    # Create CatBoost Pool objects
    train_pool = Pool(X_train, y_train, cat_features=cat_features)
    val_pool = Pool(X_val, y_val, cat_features=cat_features)

    # Prepare model parameters
    params = {
        "loss_function": "MultiClass",
        "eval_metric": "MultiClass",
        "verbose": False,
        "thread_count": n_jobs,
        **config,
    }

    # Initialize and train model
    model = CatBoostClassifier(**params)
    model.fit(
        train_pool,
        eval_set=val_pool,
        early_stopping_rounds=50,
        verbose=False,
    )

    # Make predictions
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average="weighted")

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
    """
    # MLflow setup
    if track_with_mlflow:
        if mlflow_tracking_uri:
            mlflow.set_tracking_uri(mlflow_tracking_uri)
        if experiment_name:
            mlflow.set_experiment(experiment_name)

    # Load and split data
    X_train, X_val, y_train, y_val, cat_features = load_data(data_path)

    # Define search space for CatBoost
    search_space = {
        "learning_rate": tune.loguniform(1e-4, 1e-1),
        "depth": tune.randint(4, 10),
        "l2_leaf_reg": tune.loguniform(1, 100),
        "random_strength": tune.loguniform(1e-3, 10),
        "bagging_temperature": tune.uniform(0, 1),
        "grow_policy": tune.choice(["SymmetricTree", "Depthwise", "Lossguide"]),
        "min_data_in_leaf": tune.randint(1, 50),
        "max_leaves": tune.randint(2, 64),
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
            train_catboost,
            X_train=X_train,
            X_val=X_val,
            y_train=y_train,
            y_val=y_val,
            cat_features=cat_features,
            track_with_mlflow=track_with_mlflow,
        ),
        config=search_space,
        search_alg=search_alg,
        scheduler=scheduler,
        num_samples=num_trials,
        resources_per_trial={"cpu": n_jobs},
        local_dir="./ray_results",
        name="catboost_optimization",
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
    num_trials: int = 100,
    track_with_mlflow: bool = False,
    experiment_name: Optional[str] = None,
    mlflow_tracking_uri: Optional[str] = None,
) -> None:
    """
    Main entry point using Google Fire
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
