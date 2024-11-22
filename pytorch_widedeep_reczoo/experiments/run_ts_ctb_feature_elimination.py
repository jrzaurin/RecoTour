import pickle
from typing import Any, Dict
from pathlib import Path

import pandas as pd
import catboost as ctb
from catboost import Pool
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

    # Define protected features that should be eliminated last
    protected_features = [
        "user_id",
        "item_id",
        "gender",
        "genres",
        "age",
        "occupation",
        "zipcode",
    ]
    current_features = list(X_train.columns)

    # Phase 1: Eliminate non-protected features
    while len(current_features) > len(protected_features):
        model = ctb.train(
            pool=train_data,
            params={
                "loss_function": "Logloss",
                "eval_metric": "Logloss",
                "early_stopping_rounds": 50,
                "allow_writing_files": False,
                "verbose": False,
            },
            eval_set=val_data,
        )

        y_pred = model.predict(val_data, prediction_type="Probability")[:, 1]
        y_pred_labels = (y_pred > 0.5).astype(int)
        accuracy = accuracy_score(y_val, y_pred_labels)
        f1 = f1_score(y_val, y_pred_labels)
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

        # Only consider non-protected features for elimination
        non_protected_importance = feature_importance[
            ~feature_importance["feature"].isin(protected_features)
        ]

        if len(non_protected_importance) == 0:
            break

        least_important = non_protected_importance.nsmallest(1, "importance")[
            "feature"
        ].iloc[0]
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

        print("-" * 100)
        print(
            f"Trial {trial} metrics: accuracy: {accuracy}, f1: {f1}, val_loss: {val_loss}"
        )
        print("-" * 100)

    # Phase 2: Eliminate protected features until only 2 remain
    while len(current_features) > 2:
        model = ctb.train(
            pool=train_data,
            params={
                "loss_function": "Logloss",
                "eval_metric": "Accuracy",
                "early_stopping_rounds": 50,
                "allow_writing_files": False,
                "verbose": False,
            },
            eval_set=val_data,
        )

        y_pred = model.predict(val_data, prediction_type="Probability")[:, 1]
        y_pred_labels = (y_pred > 0.5).astype(int)
        accuracy = accuracy_score(y_val, y_pred_labels)
        f1 = f1_score(y_val, y_pred_labels)
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

        print("-" * 100)
        print(
            f"Trial {trial} metrics: accuracy: {accuracy}, f1: {f1}, val_loss: {val_loss}"
        )
        print("-" * 100)

    results_dir = (
        Path(DATA_AND_ARTIFACTS_DIR) / "results" / "results_ctb_feature_elimination"
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "results.pkl", "wb") as f:
        pickle.dump(results, f)

    return results


if __name__ == "__main__":
    results = run_catboost_feature_elimination()
