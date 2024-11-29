import json
import pickle
from typing import List, Tuple, Literal
from pathlib import Path

import pandas as pd
import catboost as ctb

# import lightgbm as lgb
from sklearn.metrics import f1_score, accuracy_score

from rec_tools.constants import RESULTS_DIR
from rec_tools.prepare_experiments.prepare_ts_or_li import (
    experiment_for_catboost_with_text,
)


def train_catboost_with_text(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    y_train: pd.DataFrame,
    y_val: pd.DataFrame,
    cat_cols: List[str],
    text_col: str,
) -> Tuple[ctb.CatBoost, float, float]:
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
            "loss_function": "Logloss",
            "eval_metric": "Logloss",
            "early_stopping_rounds": 50,
            "allow_writing_files": False,
        },
        eval_set=val_pool,
    )

    cat_val_pred = model.predict(val_pool, prediction_type="Probability")[:, 1]
    cat_val_pred_labels = (cat_val_pred > 0.5).astype(int)
    acc = accuracy_score(y_val, cat_val_pred_labels)
    f1 = f1_score(y_val, cat_val_pred_labels)
    print(f"CatBoost Accuracy: {acc:.4f}")
    print(f"CatBoost F1: {f1:.4f}")
    return model, acc, f1


def main(split_type: Literal["ts", "li"] = "ts") -> None:
    train_df, val_df, cat_cols = experiment_for_catboost_with_text(split_type)

    X_train = train_df.drop("rating", axis=1)
    y_train = train_df["rating"]
    X_val = val_df.drop("rating", axis=1)
    y_val = val_df["rating"]

    results_dir = (
        Path(RESULTS_DIR) / f"results_ctb_with_text_and_default_params_{split_type}"
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    ctb_model, ctb_acc, ctb_f1 = train_catboost_with_text(
        X_train, X_val, y_train, y_val, cat_cols, "overview"  # type: ignore
    )

    with open(results_dir / "model.pkl", "wb") as f:
        pickle.dump(ctb_model, f)

    metrics = {
        "catboost": {
            "accuracy": ctb_acc,
            "f1": ctb_f1,
            "val_loss": ctb_model.get_best_score()["validation"]["Logloss"],
        },
    }

    with open(results_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    main(split_type="ts")
    main(split_type="li")
