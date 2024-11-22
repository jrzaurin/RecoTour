import pickle
from typing import Any, Dict
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import f1_score, accuracy_score

from rec_tools.constants import DATA_AND_ARTIFACTS_DIR
from rec_tools.prepare_experiments.prepare_ts import (
    prepare_experiment_with_feature_engineering,
)


def create_initial_datasets():
    train_df, val_df, cat_cols, _ = prepare_experiment_with_feature_engineering(
        use_umap="ch", gbm="lgbm"
    )

    y_train = train_df["rating"]
    y_val = val_df["rating"]
    X_train = train_df.drop("rating", axis=1)
    X_val = val_df.drop("rating", axis=1)

    train_data = lgb.Dataset(
        X_train, label=y_train, categorical_feature=cat_cols, free_raw_data=False
    )
    val_data = lgb.Dataset(
        X_val,
        label=y_val,
        categorical_feature=cat_cols,
        free_raw_data=False,
        reference=train_data,
    )

    return train_data, val_data, X_train, X_val, y_train, y_val, cat_cols


def run_lgb_feature_elimination() -> Dict[int, Dict[str, Any]]:
    train_data, val_data, X_train, X_val, y_train, y_val, cat_cols = (
        create_initial_datasets()
    )

    results = {}
    trial = 0

    params = {
        "num_iterations": 1000,
        "objective": "binary",
        "metric": "binary_logloss",
        "verbose": -1,
    }

    current_features = list(X_train.columns)

    while len(current_features) >= 2:
        model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            callbacks=[
                lgb.early_stopping(100),
            ],
        )

        y_pred = (np.array(model.predict(X_val)) > 0.5).astype(int)

        accuracy = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        val_loss = model.best_score["valid_0"]["binary_logloss"]

        results[trial] = {
            "features": current_features.copy(),
            "acc": accuracy,
            "f1": f1,
            "val_loss": val_loss,
        }

        importance = model.feature_importance(importance_type="split")
        feature_importance = pd.DataFrame(
            {"feature": current_features, "importance": importance}
        )

        # if there are features with 0 importance, remove them
        feature_importance = feature_importance[feature_importance["importance"] > 0]
        least_important = feature_importance.nsmallest(1, "importance")["feature"].iloc[
            0
        ]
        current_features.remove(least_important)

        X_train = X_train[current_features]
        X_val = X_val[current_features]

        train_data = lgb.Dataset(
            X_train,
            label=y_train,
            categorical_feature=[col for col in cat_cols if col in current_features],
            free_raw_data=False,
        )
        val_data = lgb.Dataset(
            X_val,
            label=y_val,
            categorical_feature=[col for col in cat_cols if col in current_features],
            free_raw_data=False,
            reference=train_data,
        )

        trial += 1

        print("-" * 100)
        print(
            f"Trial {trial} metrics: accuracy: {accuracy}, f1: {f1}, val_loss: {val_loss}"
        )
        print("-" * 100)

    results_dir = (
        Path(DATA_AND_ARTIFACTS_DIR) / "results" / "results_lgb_feature_elimination"
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "results.pkl", "wb") as f:
        pickle.dump(results, f)

    return results


if __name__ == "__main__":
    results = run_lgb_feature_elimination()
