import json
import pickle
from typing import Dict, Tuple, Literal
from pathlib import Path

import numpy as np
import pandas as pd
from surprise import SVD, Reader, Dataset
from sklearn.metrics import f1_score, accuracy_score, root_mean_squared_error

from rec_tools.constants import RESULTS_DIR
from rec_tools.prepare_experiments.prepare_ts_or_li import (
    experiment_without_feat_engineering,
)


def train_svd(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    binary_target: bool,
) -> Tuple[SVD, Dict[str, float]]:
    # Configure reader for ratings
    reader = Reader(rating_scale=(0, 1) if binary_target else (1, 5))

    # Convert dataframes to Surprise format
    train_data = Dataset.load_from_df(
        train_df[["user_id", "item_id", "rating"]], reader
    ).build_full_trainset()

    model = SVD()
    model.fit(train_data)

    val_predictions = [
        model.predict(uid, iid).est
        for uid, iid in zip(val_df["user_id"], val_df["item_id"])
    ]

    if binary_target:
        val_pred_labels = (np.array(val_predictions) > 0.5).astype(int)
        acc = accuracy_score(val_df["rating"], val_pred_labels)
        f1 = f1_score(val_df["rating"], val_pred_labels)
        metrics = {
            "accuracy": acc,
            "f1": f1,
        }
        print(f"SVD Accuracy: {acc:.4f}")
        print(f"SVD F1: {f1:.4f}")
    else:
        rmse = root_mean_squared_error(val_df["rating"], val_predictions)
        metrics = {"rmse": rmse}
        print(f"SVD RMSE: {rmse:.4f}")

    return model, metrics


def main(split_type: Literal["ts", "li"] = "ts", binary_target: bool = True) -> None:
    train_df, val_df, _ = experiment_without_feat_engineering(split_type, binary_target)

    results_dir = (
        Path(RESULTS_DIR)
        / f"results_svd_with_default_params_{split_type}_{'binary' if binary_target else 'regression'}"
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    svd_model, svd_metrics = train_svd(train_df, val_df, binary_target)

    with open(results_dir / "model.pkl", "wb") as f:
        pickle.dump(svd_model, f)

    with open(results_dir / "results.json", "w") as f:
        json.dump(svd_metrics, f, indent=4)


if __name__ == "__main__":
    main(split_type="ts", binary_target=True)
    main(split_type="li", binary_target=True)
    main(split_type="ts", binary_target=False)
    main(split_type="li", binary_target=False)
