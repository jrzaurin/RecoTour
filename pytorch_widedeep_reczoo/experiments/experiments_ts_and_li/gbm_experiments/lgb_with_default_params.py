import json
import pickle
from typing import Dict, List, Tuple, Literal
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import f1_score, accuracy_score, root_mean_squared_error
from pytorch_widedeep.utils import LabelEncoder

from rec_tools.constants import RESULTS_DIR
from rec_tools.prepare_experiments.prepare_ts_or_li import (
    experiment_without_feat_engineering,
)


def train_lightgbm(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: pd.DataFrame,
    y_val: pd.DataFrame,
    binary_target: bool,
    cat_cols: List[str],
) -> Tuple[lgb.Booster, Dict[str, float]]:
    train_dataset = lgb.Dataset(
        X_train, label=y_train, categorical_feature=cat_cols, free_raw_data=False
    )
    val_dataset = lgb.Dataset(
        X_val, label=y_val, reference=train_dataset, free_raw_data=False
    )

    model = lgb.train(
        {
            "n_estimators": 1000,
            "objective": "binary" if binary_target else "regression",
            "metric": "binary_logloss" if binary_target else "rmse",
        },
        train_dataset,
        valid_sets=[train_dataset, val_dataset],
        valid_names=["train", "valid"],
        callbacks=[lgb.early_stopping(50, verbose=True), lgb.log_evaluation(period=1)],
    )

    val_pred = model.predict(X_val)
    val_pred_labels = (
        (np.array(val_pred) > 0.5).astype(int) if binary_target else val_pred
    )

    if binary_target:
        acc = accuracy_score(y_val, val_pred_labels)
        f1 = f1_score(y_val, val_pred_labels)
        print(f"LightGBM Accuracy: {acc:.4f}")
        print(f"LightGBM F1: {f1:.4f}")
        return model, {"accuracy": acc, "f1": f1}
    else:
        rmse = root_mean_squared_error(y_val, val_pred)
        print(f"LightGBM RMSE: {rmse:.4f}")
        return model, {"rmse": rmse}


def main(split_type: Literal["ts", "li"] = "ts", binary_target: bool = True) -> None:
    (
        train_df,
        val_df,
        cat_cols,
    ) = experiment_without_feat_engineering(split_type, binary_target)

    encoder = LabelEncoder(columns_to_encode=cat_cols)

    train_df_encoded = encoder.fit_transform(train_df)
    val_df_encoded = encoder.transform(val_df)

    X_train = train_df_encoded.drop("rating", axis=1)
    y_train = train_df_encoded["rating"]
    X_val = val_df_encoded.drop("rating", axis=1)
    y_val = val_df_encoded["rating"]

    results_dir = (
        Path(RESULTS_DIR)
        / f"results_lgb_with_default_params_{split_type}_{'binary' if binary_target else 'regression'}"
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    model, lgb_metrics = train_lightgbm(
        X_train, X_val, y_train, y_val, binary_target=binary_target, cat_cols=cat_cols
    )

    with open(results_dir / "model.pkl", "wb") as f:
        pickle.dump(model, f)

    if binary_target:
        metrics = {
            "accuracy": lgb_metrics["accuracy"],
            "f1": lgb_metrics["f1"],
            "val_loss": model.best_score["valid"]["binary_logloss"],
        }
    else:
        metrics = {
            "rmse": lgb_metrics["rmse"],
            "val_loss": model.best_score["valid"]["rmse"],
        }

    with open(results_dir / "results.json", "w") as f:
        json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    main(split_type="ts", binary_target=True)
    main(split_type="li", binary_target=True)
    main(split_type="ts", binary_target=False)
    main(split_type="li", binary_target=False)
