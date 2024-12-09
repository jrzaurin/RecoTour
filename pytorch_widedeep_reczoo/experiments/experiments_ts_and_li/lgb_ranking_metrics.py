import pickle
from typing import Dict, List, Tuple, Literal
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
    find_categorical_cols,
    impute_categorical_cols,
    load_and_merge_features,
    experiment_without_feat_engineering,
)


def load_best_results_features_and_iteration(
    split_type: Literal["ts", "li"],
    binary_target: bool,
) -> Tuple[List[str], int]:
    res_dir = (
        Path(RESULTS_DIR)
        / f"results_lgb_with_feature_elimination_ch_{split_type}_{'binary' if binary_target else 'regression'}"
    )
    with open(res_dir / "results.pkl", "rb") as f:
        results = pickle.load(f)

    best_trial = max(results, key=lambda x: -results[x]["val_loss"])
    best_trial_features = results[best_trial]["features"]
    best_iteration = results[best_trial]["best_iteration"]

    return best_trial_features, best_iteration


def set_lgb_datasets(
    split_type: Literal["ts", "li"],
    with_feat_engineering: bool = False,
    binary_target: bool = True,
) -> Tuple[lgb.Dataset, lgb.Dataset]:
    if with_feat_engineering:
        train_df, val_df = load_and_merge_features(
            split="train_val", use_umap="ch", split_type=split_type
        )
        test_df = load_and_merge_features(
            split="test", use_umap="ch", split_type=split_type
        )
        full_train_df = pd.concat([train_df, val_df], ignore_index=True)
        _cat_cols = find_categorical_cols(full_train_df)

        full_train_df = impute_categorical_cols(full_train_df, _cat_cols)
        if binary_target:
            full_train_df = binarize_target(full_train_df)

        best_result_features, _ = load_best_results_features_and_iteration(
            split_type, binary_target
        )

        full_train_df = full_train_df[best_result_features + ["rating"]]
        cat_cols = [col for col in best_result_features if col in _cat_cols]

        test_df = test_df[best_result_features + ["rating"]]  # type: ignore
    else:
        train_df, val_df, cat_cols = experiment_without_feat_engineering(
            split_type=split_type, binary_target=binary_target
        )
        test_df = pd.read_csv(
            Path(DATA_DIR)
            / TRAIN_VAL_TEST_SPLITS_DIR
            / MOVIELENS_SPLITS_DIR
            / "test.csv"
        )
        test_df = test_df[train_df.columns]
        full_train_df = pd.concat([train_df, val_df], ignore_index=True)

    encoder = LabelEncoder(columns_to_encode=cat_cols)
    full_train_df_encoded = encoder.fit_transform(full_train_df)

    test_df = impute_categorical_cols(test_df, cat_cols)
    if binary_target:
        test_df = binarize_target(test_df)

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
    with_feat_engineering: bool = False,
    split_type: Literal["ts", "li"] = "ts",
    k_values: List[int] = [5, 10, 20],
    binary_target: bool = True,
) -> Dict[int, Dict[str, float]]:

    if with_feat_engineering:
        _, best_iteration = load_best_results_features_and_iteration(
            split_type, binary_target
        )
    else:
        with open(
            Path(RESULTS_DIR)
            / f"results_lgb_with_default_params_{split_type}_{'binary' if binary_target else 'regression'}"
            / "model.pkl",
            "rb",
        ) as f:
            model_with_default_params = pickle.load(f)
        best_iteration = model_with_default_params.num_trees()

    with_feat_engineering_suffix = "with" if with_feat_engineering else "without"
    binary_target_suffix = "binary" if binary_target else "regression"
    results_dir = (
        Path(RESULTS_DIR)
        / f"results_lgb_ranking_metrics_{split_type}_{with_feat_engineering_suffix}_{binary_target_suffix}"
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    train_data, test_data = set_lgb_datasets(
        split_type, with_feat_engineering, binary_target
    )

    model = lgb.train(
        {
            "n_estimators": best_iteration,
            "objective": "binary" if binary_target else "regression",
            "metric": "binary_logloss" if binary_target else "rmse",
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
    train_lgb_model_and_evaluate_ranking_metrics(
        with_feat_engineering=True, split_type="ts", binary_target=True
    )

    train_lgb_model_and_evaluate_ranking_metrics(
        with_feat_engineering=False, split_type="ts", binary_target=True
    )

    train_lgb_model_and_evaluate_ranking_metrics(
        with_feat_engineering=True, split_type="li", binary_target=True
    )

    train_lgb_model_and_evaluate_ranking_metrics(
        with_feat_engineering=False, split_type="li", binary_target=True
    )
