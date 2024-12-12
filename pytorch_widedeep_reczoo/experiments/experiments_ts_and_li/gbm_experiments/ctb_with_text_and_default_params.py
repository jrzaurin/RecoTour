import json
import pickle
from typing import Dict, List, Tuple, Literal
from pathlib import Path

import pandas as pd
import catboost as ctb

# import lightgbm as lgb
from sklearn.metrics import f1_score, accuracy_score, root_mean_squared_error

from rec_tools.constants import RESULTS_DIR
from rec_tools.prepare_experiments.prepare_ts_or_li import (
    experiment_for_catboost_with_text,
)


def train_catboost_with_text(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    y_train: pd.DataFrame,
    y_val: pd.DataFrame,
    binary_target: bool,
    cat_cols: List[str],
    text_col: str,
) -> Tuple[ctb.CatBoost, Dict[str, float]]:
    train_pool = ctb.Pool(
        train_df[cat_cols + [text_col]],
        label=y_train,
        cat_features=cat_cols,
        text_features=[text_col],
    )
    val_pool = ctb.Pool(
        val_df[cat_cols + [text_col]],
        label=y_val,
        cat_features=cat_cols,
        text_features=[text_col],
    )
    model = ctb.train(
        pool=train_pool,
        params={
            "loss_function": "Logloss" if binary_target else "RMSE",
            "eval_metric": "Logloss" if binary_target else "RMSE",
            "early_stopping_rounds": 50,
            "allow_writing_files": False,
        },
        eval_set=val_pool,
    )

    if binary_target:
        cat_val_pred = model.predict(val_pool, prediction_type="Probability")[:, 1]
        cat_val_pred_labels = (cat_val_pred > 0.5).astype(int)
        acc = accuracy_score(y_val, cat_val_pred_labels)
        f1 = f1_score(y_val, cat_val_pred_labels)
        metrics = {
            "accuracy": acc,
            "f1": f1,
            "val_loss": model.get_best_score()["validation"]["Logloss"],
        }
        print(f"CatBoost Accuracy: {acc:.4f}")
        print(f"CatBoost F1: {f1:.4f}")
    else:
        rmse = root_mean_squared_error(y_val, model.predict(val_pool))
        metrics = {
            "rmse": rmse,
            "val_loss": model.get_best_score()["validation"]["RMSE"],
        }
        print(f"CatBoost RMSE: {rmse:.4f}")

    return model, metrics


def main(split_type: Literal["ts", "li"] = "ts", binary_target: bool = True) -> None:
    train_df, val_df, cat_cols = experiment_for_catboost_with_text(
        split_type, binary_target
    )

    X_train = train_df.drop("rating", axis=1)
    y_train = train_df["rating"]
    X_val = val_df.drop("rating", axis=1)
    y_val = val_df["rating"]

    results_dir = (
        Path(RESULTS_DIR)
        / f"results_ctb_with_text_and_default_params_{split_type}_{'binary' if binary_target else 'regression'}"
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    ctb_model, metrics = train_catboost_with_text(
        X_train, X_val, y_train, y_val, binary_target, cat_cols, "overview"
    )

    with open(results_dir / "model.pkl", "wb") as f:
        pickle.dump(ctb_model, f)

    with open(results_dir / "results.json", "w") as f:
        json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    main(split_type="ts", binary_target=True)
    main(split_type="li", binary_target=True)
    main(split_type="ts", binary_target=False)
    main(split_type="li", binary_target=False)
