import pickle
from typing import Dict, List, Tuple
from pathlib import Path

import pandas as pd
import catboost as ctb

from rec_tools.constants import (
    DATA_DIR,
    RESULTS_DIR,
    MOVIELENS_SPLITS_DIR,
    TRAIN_VAL_TEST_SPLITS_DIR,
)
from rec_tools.ranking_metrics import map_at_k, hit_ratio_at_k, binary_ndcg_at_k
from rec_tools.prepare_experiments.prepare_ts_or_li import (
    binarize_target,
    impute_categorical_cols,
    experiment_without_feat_engineering,
)


def set_ctb_datasets() -> Tuple[ctb.Pool, ctb.Pool]:
    train_df, val_df, cat_cols = experiment_without_feat_engineering()

    full_train_df = pd.concat([train_df, val_df], ignore_index=True)

    test_df = pd.read_csv(
        Path(DATA_DIR) / TRAIN_VAL_TEST_SPLITS_DIR / MOVIELENS_SPLITS_DIR / "test.csv"
    )
    test_df = binarize_target(test_df)
    test_df = impute_categorical_cols(test_df, cat_cols)

    full_train_df = full_train_df[cat_cols + ["rating"]]
    test_df = test_df[cat_cols + ["rating"]]

    X_train = full_train_df.drop(columns=["rating"])
    y_train = full_train_df["rating"]
    X_test = test_df.drop(columns=["rating"])
    y_test = test_df["rating"]

    train_data = ctb.Pool(
        X_train,
        label=y_train,
        cat_features=cat_cols,
    )

    test_data = ctb.Pool(
        X_test,
        label=y_test,
        cat_features=cat_cols,
    )

    return train_data, test_data


def train_ctb_model_and_evaluate_ranking_metrics(
    k_values: List[int] = [5, 10, 20]
) -> Dict[int, Dict[str, float]]:

    with open(
        Path(RESULTS_DIR) / "results_ctb_with_default_params" / "model.pkl", "rb"
    ) as f:
        model_with_default_params = pickle.load(f)

    results_dir = Path(RESULTS_DIR) / "ctb_ranking_metrics"
    results_dir.mkdir(parents=True, exist_ok=True)

    train_data, test_data = set_ctb_datasets()

    model = ctb.train(
        pool=train_data,
        params={
            "loss_function": "Logloss",
            "iterations": model_with_default_params.get_best_iteration(),
            "allow_writing_files": False,
            "verbose": True,
        },
    )

    y_pred = model.predict(test_data, prediction_type="Probability")[:, 1]
    y_test = test_data.get_label()

    results: Dict[int, Dict[str, float]] = {}
    for k in k_values:
        test_ndcg = binary_ndcg_at_k(y_pred, y_test, n_items=100, k=k)
        test_map = map_at_k(y_pred, y_test, n_items=100, k=k)
        test_hr = hit_ratio_at_k(y_pred, y_test, n_items=100, k=k)

        results[k] = {
            "ndcg": test_ndcg,
            "map": test_map,
            "hr": test_hr,
        }

        print(f"NDCG@{k}: {test_ndcg}")
        print(f"MAP@{k}: {test_map}")
        print(f"HR@{k}: {test_hr}")

    with open(results_dir / "results.pkl", "wb") as f:
        pickle.dump(results, f)

    return results


if __name__ == "__main__":
    train_ctb_model_and_evaluate_ranking_metrics()
