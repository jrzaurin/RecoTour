import pickle
from typing import Dict, List, Tuple, Literal
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
    find_categorical_cols,
    impute_categorical_cols,
    load_and_merge_features,
    experiment_without_feat_engineering,
)

SELECT_FEATURES_ALGORITHM_SUFFIX_MAP = {
    "RecursiveByPredictionValuesChange": "pvc",
    "RecursiveByLossFunctionChange": "lfc",
    "RecursiveByShapValues": "shap",
}


def load_best_results_features_and_iteration(
    split_type: Literal["ts", "li"],
    binary_target: bool,
    select_features_algorithm: Literal[
        "RecursiveByLossFunctionChange",
        "RecursiveByShapValues",
        "RecursiveByPredictionValuesChange",
    ],
) -> Tuple[List[str], int]:

    fs_suffix = SELECT_FEATURES_ALGORITHM_SUFFIX_MAP[select_features_algorithm]
    res_dir = (
        Path(RESULTS_DIR)
        / f"results_ctb_ranking_metrics_{split_type}_{'binary' if binary_target else 'regression'}_{fs_suffix}"
    )
    with open(res_dir / "results.pkl", "rb") as f:
        results = pickle.load(f)

    best_trial_features = results["features"]
    best_iteration = results["best_iteration"]

    return best_trial_features, best_iteration


def set_ctb_datasets(
    split_type: Literal["ts", "li"],
    with_feat_engineering: bool = False,
    binary_target: bool = True,
    select_features_algorithm: (
        Literal[
            "RecursiveByLossFunctionChange",
            "RecursiveByShapValues",
            "RecursiveByPredictionValuesChange",
        ]
        | None
    ) = None,
) -> Tuple[ctb.Pool, ctb.Pool]:
    if with_feat_engineering:
        train_df, val_df = load_and_merge_features(
            split="train_val",
            use_umap="ch",
            split_type=split_type,
        )
        test_df = load_and_merge_features(
            split="test",
            use_umap="ch",
            split_type=split_type,
        )
        full_train_df = pd.concat([train_df, val_df], ignore_index=True)

        _cat_cols = find_categorical_cols(full_train_df)
        full_train_df = impute_categorical_cols(full_train_df, _cat_cols)
        if binary_target:
            full_train_df = binarize_target(full_train_df)
        best_result_features, _ = load_best_results_features_and_iteration(
            split_type, binary_target, select_features_algorithm
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

    test_df = impute_categorical_cols(test_df, cat_cols)
    if binary_target:
        test_df = binarize_target(test_df)

    X_train = full_train_df.drop(columns=["rating"])
    y_train = full_train_df["rating"]
    X_test = test_df.drop(columns=["rating"])
    y_test = test_df["rating"]

    train_data = ctb.Pool(data=X_train, label=y_train, cat_features=cat_cols)
    test_data = ctb.Pool(data=X_test, label=y_test, cat_features=cat_cols)

    return train_data, test_data


def train_ctb_model_and_evaluate_ranking_metrics(
    split_type: Literal["ts", "li"] = "ts",
    with_feat_engineering: bool = False,
    binary_target: bool = True,
    k_values: List[int] = [5, 10, 20],
    select_features_algorithm: (
        Literal[
            "RecursiveByLossFunctionChange",
            "RecursiveByShapValues",
            "RecursiveByPredictionValuesChange",
        ]
        | None
    ) = None,
) -> Dict[int, Dict[str, float]]:

    if with_feat_engineering:
        _, best_iteration = load_best_results_features_and_iteration(
            split_type,
            binary_target,
            select_features_algorithm,
        )
    else:
        with open(
            Path(RESULTS_DIR)
            / f"results_ctb_with_default_params_{split_type}_{'binary' if binary_target else 'regression'}"
            / "model.pkl",
            "rb",
        ) as f:
            model_with_default_params = pickle.load(f)
        best_iteration = model_with_default_params.tree_count_

    with_feat_engineering_suffix = "with" if with_feat_engineering else "without"
    fs_suffix = (
        SELECT_FEATURES_ALGORITHM_SUFFIX_MAP[select_features_algorithm]
        if select_features_algorithm
        else "nofs"
    )
    binary_suffix = "binary" if binary_target else "regression"
    results_dir = (
        Path(RESULTS_DIR)
        / f"results_ctb_ranking_metrics_{split_type}_{with_feat_engineering_suffix}_{fs_suffix}_{binary_suffix}"
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    train_data, test_data = set_ctb_datasets(
        split_type, with_feat_engineering, binary_target, select_features_algorithm
    )

    model = ctb.train(
        pool=train_data,
        params={
            "iterations": best_iteration,
            "loss_function": "Logloss" if binary_target else "RMSE",
            "eval_metric": "Logloss" if binary_target else "RMSE",
            "verbose": False,
        },
    )

    if binary_target:
        y_pred = model.predict(test_data, prediction_type="Probability")[:, 1]
    else:
        y_pred = model.predict(test_data)
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
    train_ctb_model_and_evaluate_ranking_metrics(
        with_feat_engineering=True,
        split_type="ts",
        binary_target=True,
        select_features_algorithm="RecursiveByShapValues",
    )
    train_ctb_model_and_evaluate_ranking_metrics(
        with_feat_engineering=True,
        split_type="ts",
        binary_target=False,
        select_features_algorithm="RecursiveByShapValues",
    )

    train_ctb_model_and_evaluate_ranking_metrics(
        with_feat_engineering=True,
        split_type="li",
        binary_target=True,
        select_features_algorithm="RecursiveByShapValues",
    )
    train_ctb_model_and_evaluate_ranking_metrics(
        with_feat_engineering=True,
        split_type="li",
        binary_target=False,
        select_features_algorithm="RecursiveByShapValues",
    )

    train_ctb_model_and_evaluate_ranking_metrics(
        with_feat_engineering=False,
        split_type="ts",
        binary_target=True,
    )
    train_ctb_model_and_evaluate_ranking_metrics(
        with_feat_engineering=False,
        split_type="ts",
        binary_target=False,
    )

    train_ctb_model_and_evaluate_ranking_metrics(
        with_feat_engineering=False,
        split_type="li",
        binary_target=True,
    )
    train_ctb_model_and_evaluate_ranking_metrics(
        with_feat_engineering=False,
        split_type="li",
        binary_target=False,
    )
