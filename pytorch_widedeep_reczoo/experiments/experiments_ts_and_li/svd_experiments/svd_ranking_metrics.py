import json
import pickle
from typing import Any, Dict, List, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
from surprise import SVD, Reader, Dataset

from rec_tools.constants import (
    DATA_DIR,
    RESULTS_DIR,
    MOVIELENS_SPLITS_DIR,
    TRAIN_VAL_TEST_SPLITS_DIR,
)
from rec_tools.ranking_metrics import map_at_k, hit_ratio_at_k, binary_ndcg_at_k
from rec_tools.prepare_experiments.prepare_ts_or_li import binarize_target


def load_train_and_test_datasets() -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(
        Path(DATA_DIR)
        / TRAIN_VAL_TEST_SPLITS_DIR
        / MOVIELENS_SPLITS_DIR
        / "full_train.csv"
    )
    test_df = pd.read_csv(
        Path(DATA_DIR) / TRAIN_VAL_TEST_SPLITS_DIR / MOVIELENS_SPLITS_DIR / "test.csv"
    )

    return train_df, test_df


def best_experiment_name():
    binary_results_dir = Path(RESULTS_DIR) / "binary_results"
    regression_results_dir = Path(RESULTS_DIR) / "regression_results"

    binary_results_df = pd.read_csv(binary_results_dir / "binary_metrics.csv")
    regression_results_df = pd.read_csv(
        regression_results_dir / "regression_metrics.csv"
    )
    binary_results_df_svd = binary_results_df[
        binary_results_df.experiment.str.contains("svd")
    ]
    regression_results_df_svd = regression_results_df[
        regression_results_df.experiment.str.contains("svd")
    ]

    best_exp_name_dict = {}
    for experiment_type in ["binary", "regression"]:
        _df = (
            binary_results_df_svd
            if experiment_type == "binary"
            else regression_results_df_svd
        )
        best_exp_name_dict[experiment_type] = {}
        for split_type in ["ts", "li"]:
            _split_type = f"_{split_type}_"
            _df_split_type = _df[_df["experiment"].str.contains(_split_type)]
            _df_split_type = _df_split_type.sort_values(by="val_loss", ascending=True)
            best_exp_name_dict[experiment_type][split_type] = _df_split_type[
                "experiment"
            ].iloc[0]

    return best_exp_name_dict


def load_info_params_with_hyperopt(experiment_name: str) -> Dict[str, Any]:
    with open(
        Path(RESULTS_DIR) / experiment_name / "results.json",
        "r",
    ) as f:
        results = json.load(f)
    return results[
        "best_params"
    ]  # TODO: change this so is consistent with the other ones (i.e. 'config')


def train_svd_model_and_evaluate_ranking_metrics(
    experiment_name: str,
    k_values: List[int] = [5, 10, 20],
    binary_target: bool = True,
):

    results_dir = (
        Path(RESULTS_DIR) / "results_svd_ranking_metrics" / f"{experiment_name}"
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    train_df, test_df = load_train_and_test_datasets()
    test_df = binarize_target(test_df)
    if binary_target:
        train_df = binarize_target(train_df)

    reader = Reader(rating_scale=(0, 1) if binary_target else (1, 5))
    train_data = Dataset.load_from_df(
        train_df[["user_id", "item_id", "rating"]], reader
    ).build_full_trainset()

    if "hyperopt" in experiment_name:
        model = SVD(**load_info_params_with_hyperopt(experiment_name))
    else:
        model = SVD()

    model.fit(train_data)

    y_test = test_df["rating"].values
    y_pred = np.array(
        [
            model.predict(uid, iid).est
            for uid, iid in zip(test_df["user_id"], test_df["item_id"])
        ]
    )

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
    experiments_names = best_experiment_name()

    # {
    #   'binary':
    #    {'ts': 'results_svd_with_hyperopt_ts_binary',
    #     'li': 'results_svd_with_hyperopt_li_binary'},
    #   'regression':
    #    {'ts': 'results_svd_with_hyperopt_ts_regression',
    #     'li': 'results_svd_with_hyperopt_li_regression'}
    # }

    # with hyperopt
    for experiment_type in ["binary", "regression"]:
        for split_type in ["ts", "li"]:
            print("-" * 100)
            print(f"Experiment: {experiments_names[experiment_type][split_type]}")
            print("-" * 100)
            experiment_name = experiments_names[experiment_type][split_type]
            train_svd_model_and_evaluate_ranking_metrics(
                experiment_name,
                k_values=[5, 10, 20],
                binary_target=experiment_type == "binary",
            )

    # without hyperopt
    default_params_experiments_names: Dict[str, Dict[str, str]] = {}
    default_params_experiments_names["binary"] = {}
    default_params_experiments_names["regression"] = {}
    default_params_experiments_names["binary"][
        "ts"
    ] = "results_svd_with_default_params_ts_binary"
    default_params_experiments_names["regression"][
        "ts"
    ] = "results_svd_with_default_params_ts_regression"
    default_params_experiments_names["binary"][
        "li"
    ] = "results_svd_with_default_params_li_binary"
    default_params_experiments_names["regression"][
        "li"
    ] = "results_svd_with_default_params_li_regression"
    for experiment_type in ["binary", "regression"]:
        for split_type in ["ts", "li"]:
            print("-" * 100)
            print(
                f"Experiment: {default_params_experiments_names[experiment_type][split_type]}"
            )
            print("-" * 100)
            experiment_name = default_params_experiments_names[experiment_type][
                split_type
            ]
            train_svd_model_and_evaluate_ranking_metrics(
                experiment_name,
                k_values=[5, 10, 20],
                binary_target=experiment_type == "binary",
            )
