import pickle
from typing import Any, Dict, List, Tuple, Literal
from pathlib import Path

import pandas as pd
import catboost as ctb
from catboost import Pool
from sklearn.metrics import f1_score, accuracy_score, root_mean_squared_error

from rec_tools.constants import RESULTS_DIR
from rec_tools.prepare_experiments.prepare_ts_or_li import (
    experiment_with_feat_engineering,
)


def create_initial_datasets(
    use_umap: Literal["st", "ch"],
    split_type: Literal["ts", "li"],
    binary_target: bool,
) -> Tuple[Pool, Pool, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str]]:
    train_df, val_df, cat_cols = experiment_with_feat_engineering(
        use_umap, split_type, binary_target
    )

    y_train = train_df["rating"]
    y_val = val_df["rating"]
    X_train = train_df.drop("rating", axis=1)
    X_val = val_df.drop("rating", axis=1)

    train_data = Pool(X_train, label=y_train, cat_features=cat_cols)
    val_data = Pool(X_val, label=y_val, cat_features=cat_cols)

    return train_data, val_data, X_train, X_val, y_train, y_val, cat_cols


def run_catboost_feature_elimination(
    use_umap: Literal["st", "ch"],
    split_type: Literal["ts", "li"],
    binary_target: bool,
) -> Dict[int, Dict[str, Any]]:
    train_data, val_data, X_train, X_val, y_train, y_val, cat_cols = (
        create_initial_datasets(use_umap, split_type, binary_target)
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
                "num_boost_round": 500,
                "loss_function": "Logloss" if binary_target else "RMSE",
                "eval_metric": "Logloss" if binary_target else "RMSE",
                "early_stopping_rounds": 50,
                "allow_writing_files": False,
                "verbose": True,
            },
            eval_set=val_data,
        )

        if binary_target:
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
        else:
            rmse = root_mean_squared_error(y_val, model.predict(val_data))
            val_loss = model.get_best_score()["validation"]["RMSE"]
            results[trial] = {
                "features": current_features.copy(),
                "rmse": rmse,
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
        if binary_target:
            print(
                f"Trial {trial} metrics: accuracy: {accuracy}, f1: {f1}, val_loss: {val_loss}"
            )
        else:
            print(f"Trial {trial} metrics: rmse: {rmse}, val_loss: {val_loss}")
        print("-" * 100)

    # Phase 2: Eliminate protected features until only 2 remain
    # Note: there is a lot of code repetition. For now I am happy to leave it as
    # is.
    while len(current_features) > 2:
        model = ctb.train(
            pool=train_data,
            params={
                "loss_function": "Logloss" if binary_target else "RMSE",
                "eval_metric": "Logloss" if binary_target else "RMSE",
                "early_stopping_rounds": 50,
                "allow_writing_files": False,
                "verbose": False,
            },
            eval_set=val_data,
        )

        if binary_target:
            y_pred = model.predict(val_data, prediction_type="Probability")[:, 1]
            y_pred_labels = (y_pred > 0.5).astype(int)
            accuracy = accuracy_score(y_val, y_pred_labels)
            f1 = f1_score(y_val, y_pred_labels)
            val_loss = model.get_best_score()["validation"]["Logloss"]
            results[trial] = {
                "features": current_features.copy(),
                "accuracy": accuracy,
                "f1": f1,
                "val_loss": val_loss,
                "best_iteration": model.get_best_iteration(),
            }
        else:
            rmse = root_mean_squared_error(y_val, model.predict(val_data))
            val_loss = model.get_best_score()["validation"]["RMSE"]
            results[trial] = {
                "features": current_features.copy(),
                "rmse": rmse,
                "val_loss": val_loss,
                "best_iteration": model.get_best_iteration(),
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
        if binary_target:
            print(
                f"Trial {trial} metrics: accuracy: {accuracy}, f1: {f1}, val_loss: {val_loss}"
            )
        else:
            print(f"Trial {trial} metrics: rmse: {rmse}, val_loss: {val_loss}")
        print("-" * 100)

    results_dir = (
        Path(RESULTS_DIR)
        / f"results_ctb_with_feature_elimination_{use_umap}_{split_type}_{'binary' if binary_target else 'regression'}"
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "results.pkl", "wb") as f:
        pickle.dump(results, f)

    return results


if __name__ == "__main__":
    results_ts = run_catboost_feature_elimination(
        use_umap="ch", split_type="ts", binary_target=False
    )
    results_li = run_catboost_feature_elimination(
        use_umap="ch", split_type="li", binary_target=False
    )
    results_ts_binary = run_catboost_feature_elimination(
        use_umap="ch", split_type="ts", binary_target=True
    )
    results_li_binary = run_catboost_feature_elimination(
        use_umap="ch", split_type="li", binary_target=True
    )
