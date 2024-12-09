import json
import pickle
from typing import Any, Dict, List, Tuple, Literal, cast
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


def best_experiment_name():
    binary_results_dir = Path(RESULTS_DIR) / "binary_results"
    regression_results_dir = Path(RESULTS_DIR) / "regression_results"

    binary_results_df = pd.read_csv(binary_results_dir / "binary_metrics.csv")
    regression_results_df = pd.read_csv(
        regression_results_dir / "regression_metrics.csv"
    )
    binary_results_df_ctb = binary_results_df[
        binary_results_df.experiment.str.contains("ctb")
    ]
    regression_results_df_ctb = regression_results_df[
        regression_results_df.experiment.str.contains("ctb")
    ]

    best_exp_name_dict = {}
    for experiment_type in ["binary", "regression"]:
        _df = (
            binary_results_df_ctb
            if experiment_type == "binary"
            else regression_results_df_ctb
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


def load_info_best_results_features_and_iteration(
    experiment_name: str,
) -> Tuple[List[str], int]:
    res_dir = Path(RESULTS_DIR) / experiment_name
    with open(res_dir / "results.pkl", "rb") as f:
        results = pickle.load(f)

    best_trial = max(results, key=lambda x: -results[x]["val_loss"])
    best_trial_features = results[best_trial]["features"]
    best_iteration = results[best_trial]["best_iteration"]

    return best_trial_features, best_iteration


def load_info_params_with_hyperopt(experiment_name: str) -> Dict[str, Any]:
    with open(
        Path(RESULTS_DIR) / experiment_name / "results.json",
        "r",
    ) as f:
        results = json.load(f)
    return results[
        "best_params"
    ]  # TODO: change this so is consistent with the other ones (i.e. 'config')


def set_ctb_datasets_without_feat_engineering(
    split_type: Literal["ts", "li"],
    binary_target: bool = True,
) -> Tuple[ctb.Pool, ctb.Pool]:

    train_df, val_df, cat_cols = experiment_without_feat_engineering(
        split_type=split_type, binary_target=binary_target
    )

    test_df = pd.read_csv(
        Path(DATA_DIR) / TRAIN_VAL_TEST_SPLITS_DIR / MOVIELENS_SPLITS_DIR / "test.csv"
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


def set_catboost_datasets_with_feat_engineering(
    split_type: Literal["ts", "li"],
    binary_target: bool = True,
    experiment_name: str | None = None,
) -> Tuple[ctb.Pool, ctb.Pool]:

    train_df, val_df = load_and_merge_features(
        split="train_val", use_umap="ch", split_type=split_type
    )
    test_df = load_and_merge_features(
        split="test", use_umap="ch", split_type=split_type
    )
    full_train_df = pd.concat([train_df, val_df], ignore_index=True)
    cat_cols = find_categorical_cols(full_train_df)

    full_train_df = impute_categorical_cols(full_train_df, cat_cols)
    if binary_target:
        full_train_df = binarize_target(full_train_df)

    if experiment_name is not None:
        best_result_features, _ = load_info_best_results_features_and_iteration(
            experiment_name
        )
    else:
        best_result_features = [
            c for c in full_train_df.columns.tolist() if c != "rating"
        ]

    full_train_df = full_train_df[best_result_features + ["rating"]]
    cat_cols = [col for col in best_result_features if col in cat_cols]

    test_df = test_df[best_result_features + ["rating"]]  # type: ignore
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
    experiment_name: str,
    with_feat_engineering: bool = False,
    split_type: Literal["ts", "li"] = "ts",
    k_values: List[int] = [5, 10, 20],
    binary_target: bool = True,
) -> Dict[int, Dict[str, float]]:

    results_dir = Path(RESULTS_DIR) / "results_ctb_ranking_metrics"

    params: Dict[str, Any] = {}
    if with_feat_engineering:
        if "with_feature_elimination" in experiment_name:
            _, best_iteration = load_info_best_results_features_and_iteration(
                experiment_name
            )
            params["iterations"] = best_iteration
            train_data, test_data = set_catboost_datasets_with_feat_engineering(
                split_type,
                binary_target,
                experiment_name,
            )
        else:  # it will be "hyperopt"
            params = load_info_params_with_hyperopt(experiment_name)
            train_data, test_data = set_catboost_datasets_with_feat_engineering(
                split_type,
                binary_target,
            )
    else:
        with open(
            Path(RESULTS_DIR) / experiment_name / "model.pkl",
            "rb",
        ) as f:
            model_with_default_params = pickle.load(f)
        best_iteration = model_with_default_params.tree_count_
        params["iterations"] = best_iteration
        train_data, test_data = set_ctb_datasets_without_feat_engineering(
            split_type, binary_target
        )

    results_full_path = results_dir / f"{experiment_name}"
    results_full_path.mkdir(parents=True, exist_ok=True)

    params["loss_function"] = "Logloss" if binary_target else "RMSE"
    params["eval_metric"] = "Logloss" if binary_target else "RMSE"

    model = ctb.train(
        params=params,
        dtrain=train_data,
    )

    y_pred = (
        model.predict(test_data, prediction_type="Probability")[:, 1]
        if binary_target
        else model.predict(test_data)
    )
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

    with open(results_full_path / "results.pkl", "wb") as f:
        pickle.dump(results, f)

    return results


if __name__ == "__main__":

    SplitType = Literal["ts", "li"]

    experiments_names = best_experiment_name()
    # {
    #     "binary": {
    #         "ts": "results_ctb_with_feature_elimination_ch_ts_binary",
    #         "li": "results_ctb_with_feature_elimination_ch_li_binary",
    #     },
    #     "regression": {
    #         "ts": "results_ctb_with_feature_elimination_ch_ts_regression",
    #         "li": "results_ctb_with_hyperopt_ch_li_regression",
    #     },
    # }

    # with feature engineering
    for experiment_type in ["binary", "regression"]:
        for split_type in ["ts", "li"]:
            print("-" * 100)
            print(f"Experiment: {experiments_names[experiment_type][split_type]}")
            print("-" * 100)
            split_type_ = cast(SplitType, split_type)
            experiment_name = experiments_names[experiment_type][split_type]
            train_ctb_model_and_evaluate_ranking_metrics(
                with_feat_engineering=True,
                split_type=split_type_,
                binary_target=experiment_type == "binary",
                experiment_name=experiment_name,
            )

    # without feature engineering
    default_params_experiments_names: Dict[str, Dict[str, str]] = {}
    default_params_experiments_names["binary"] = {}
    default_params_experiments_names["regression"] = {}
    default_params_experiments_names["binary"][
        "ts"
    ] = "results_ctb_with_default_params_ts_binary"
    default_params_experiments_names["regression"][
        "ts"
    ] = "results_ctb_with_default_params_ts_regression"
    default_params_experiments_names["binary"][
        "li"
    ] = "results_ctb_with_default_params_li_binary"
    default_params_experiments_names["regression"][
        "li"
    ] = "results_ctb_with_default_params_li_regression"
    for experiment_type in ["binary", "regression"]:
        for split_type in ["ts", "li"]:
            print("-" * 100)
            print(
                f"Experiment: {default_params_experiments_names[experiment_type][split_type]}"
            )
            print("-" * 100)
            split_type_ = cast(SplitType, split_type)
            experiment_name = default_params_experiments_names[experiment_type][
                split_type
            ]
            train_ctb_model_and_evaluate_ranking_metrics(
                with_feat_engineering=False,
                split_type=split_type_,
                binary_target=True,
                experiment_name=experiment_name,
            )
