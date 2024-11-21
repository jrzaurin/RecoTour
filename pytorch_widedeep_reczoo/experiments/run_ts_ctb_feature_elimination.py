import pickle
from typing import Any, Dict
from pathlib import Path

import pandas as pd
from catboost import Pool, CatBoost
from sklearn.metrics import f1_score, accuracy_score

from rec_tools.constants import DATA_AND_ARTIFACTS_DIR
from rec_tools.prepare_experiments.prepare_ts import (
    prepare_experiment_with_feature_engineering,
)


def create_initial_datasets():
    train_df, val_df, cat_cols, _ = prepare_experiment_with_feature_engineering(
        use_umap="ch", gbm="catboost"
    )

    y_train = train_df["rating"]
    y_val = val_df["rating"]
    X_train = train_df.drop("rating", axis=1)
    X_val = val_df.drop("rating", axis=1)

    train_data = Pool(X_train, label=y_train, cat_features=cat_cols)
    val_data = Pool(X_val, label=y_val, cat_features=cat_cols)

    return train_data, val_data, X_train, X_val, y_train, y_val, cat_cols


def run_catboost_feature_elimination() -> Dict[int, Dict[str, Any]]:
    train_data, val_data, X_train, X_val, y_train, y_val, cat_cols = (
        create_initial_datasets()
    )

    results = {}
    trial = 0

    params = {
        "iterations": 500,
        "loss_function": "Logloss",
        "eval_metric": "Logloss",
        "random_seed": 42,
        "early_stopping_rounds": 50,
        "verbose": False,
    }

    current_features = list(X_train.columns)

    while len(current_features) >= 2:
        model = CatBoost(params)
        model.fit(train_data, eval_set=val_data, verbose=False)

        y_pred = (model.predict(X_val) > 0.5).astype(int)

        accuracy = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        val_loss = model.get_best_score()["validation"]["Logloss"]

        results[trial] = {
            "features": current_features.copy(),
            "acc": accuracy,
            "f1": f1,
            "val_loss": val_loss,
        }

        importance = model.get_feature_importance()
        feature_importance = pd.DataFrame(
            {"feature": current_features, "importance": importance}
        )

        least_important = feature_importance.nsmallest(1, "importance")["feature"].iloc[
            0
        ]
        current_features.remove(least_important)

        X_train = X_train[current_features]
        X_val = X_val[current_features]

        train_data = Pool(
            X_train,
            label=y_train,
            cat_features=[col for col in cat_cols if col in current_features],
        )
        val_data = Pool(
            X_val,
            label=y_val,
            cat_features=[col for col in cat_cols if col in current_features],
        )

        trial += 1

    results_dir = (
        Path(DATA_AND_ARTIFACTS_DIR) / "results" / "results_ctb_feature_elimination"
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "results.pkl", "wb") as f:
        pickle.dump(results, f)

    return results


if __name__ == "__main__":
    results = run_catboost_feature_elimination()
