import pickle
from typing import Dict, List, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from pytorch_widedeep.utils import LabelEncoder

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


def set_lgb_datasets() -> Tuple[lgb.Dataset, lgb.Dataset]:
    train_df, val_df, cat_cols = experiment_without_feat_engineering()
    full_train_df = pd.concat([train_df, val_df], ignore_index=True)

    test_df = pd.read_csv(
        Path(DATA_DIR) / TRAIN_VAL_TEST_SPLITS_DIR / MOVIELENS_SPLITS_DIR / "test.csv"
    )
    test_df = binarize_target(test_df)
    test_df = impute_categorical_cols(test_df, cat_cols)

    full_train_df = full_train_df[cat_cols + ["rating"]]
    test_df = test_df[cat_cols + ["rating"]]

    encoder = LabelEncoder(columns_to_encode=cat_cols)
    full_train_df_encoded = encoder.fit_transform(full_train_df)
    test_df_encoded = encoder.transform(test_df)

    X_train = full_train_df_encoded.drop(columns=["rating"])
    y_train = full_train_df["rating"]
    X_test = test_df_encoded.drop(columns=["rating"])
    y_test = test_df["rating"]

    train_data = lgb.Dataset(
        X_train,
        label=y_train,
        categorical_feature=cat_cols,
        free_raw_data=False,
    )

    test_data = lgb.Dataset(
        X_test,
        label=y_test,
        reference=train_data,
        free_raw_data=False,
    )

    return train_data, test_data


def train_lgb_model_and_evaluate_ranking_metrics(
    k_values: List[int] = [5, 10, 20]
) -> Dict[int, Dict[str, float]]:

    with open(
        Path(RESULTS_DIR) / "results_lgb_with_default_params" / "model.pkl", "rb"
    ) as f:
        model_with_default_params = pickle.load(f)

    results_dir = Path(RESULTS_DIR) / "lgb_ranking_metrics"
    results_dir.mkdir(parents=True, exist_ok=True)

    train_data, test_data = set_lgb_datasets()

    model = lgb.train(
        {
            "n_estimators": model_with_default_params.num_trees(),
            "objective": "binary",
            "metric": "binary_logloss",
        },
        train_data,
    )

    y_pred = model.predict(test_data.data)
    y_test: np.ndarray = test_data.label.values  # type: ignore

    results: Dict[int, Dict[str, float]] = {}
    for k in k_values:
        test_ndcg = binary_ndcg_at_k(y_pred, y_test, n_items=100, k=k)  # type: ignore
        test_map = map_at_k(y_pred, y_test, n_items=100, k=k)  # type: ignore
        test_hr = hit_ratio_at_k(y_pred, y_test, n_items=100, k=k)  # type: ignore

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
    train_lgb_model_and_evaluate_ranking_metrics()
