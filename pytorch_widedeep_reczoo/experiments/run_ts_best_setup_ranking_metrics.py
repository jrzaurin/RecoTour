import pickle
from typing import Dict, List, Tuple, Literal
from pathlib import Path

import pandas as pd
import catboost as ctb
import lightgbm as lgb
from pytorch_widedeep.utils import LabelEncoder

from rec_tools.constants import DATA_AND_ARTIFACTS_DIR
from rec_tools.ranking_metrics import map_at_k, hit_ratio_at_k, binary_ndcg_at_k
from rec_tools.prepare_experiments.prepare_ts import (
    load_splits,
    binarize_target,
    load_and_merge_test_data,
    impute_float_categorical_cols,
    prepare_experiment_with_feature_engineering,
    prepare_experiment_without_feat_engineering,
)


def load_results(gbm: Literal["ctb", "lgb"]) -> Dict[int, Dict[str, List[str] | float]]:
    results_path = (
        Path(DATA_AND_ARTIFACTS_DIR) / "results" / f"results_{gbm}_feature_elimination"
    )
    with open(results_path / "results.pkl", "rb") as f:
        return pickle.load(f)


def get_features(
    gbm: Literal["ctb", "lgb"],
    metric: Literal["acc", "f1", "val_loss"],
) -> pd.DataFrame:
    results = load_results(gbm)
    mode = "min" if metric == "val_loss" else "max"

    # Find best setup based on metric value
    best_setup_id = (
        max(results.keys(), key=lambda k: results[k][metric])
        if mode == "max"
        else min(results.keys(), key=lambda k: results[k][metric])
    )

    return results[best_setup_id]["features"]


def set_ctb_datasets() -> Tuple[ctb.Pool, ctb.Pool]:
    # Maybe I should be consistent and name gbm as ctb
    train_df, val_df, cat_cols, _ = prepare_experiment_with_feature_engineering(
        use_umap="ch", gbm="catboost"
    )
    full_train_df = pd.concat([train_df, val_df], ignore_index=True)
    test_df = load_and_merge_test_data(use_umap="ch")
    test_df = binarize_target(test_df)
    test_df = impute_float_categorical_cols(test_df, cat_cols)

    final_features = get_features(gbm="ctb", metric="val_loss")
    final_cat_cols = [col for col in cat_cols if col in final_features]

    full_train_df = full_train_df[final_features + ["rating"]]
    test_df = test_df[final_features + ["rating"]]

    X_train = full_train_df.drop(columns=["rating"])
    y_train = full_train_df["rating"]
    X_test = test_df.drop(columns=["rating"])
    y_test = test_df["rating"]

    train_data = ctb.Pool(
        X_train,
        label=y_train,
        cat_features=final_cat_cols,
    )

    test_data = ctb.Pool(
        X_test,
        label=y_test,
        cat_features=final_cat_cols,
    )

    return train_data, test_data


def set_lgb_datasets() -> Tuple[lgb.Dataset, lgb.Dataset]:
    # Maybe I should be consistent and name gbm as lgb
    train_df, val_df = load_splits()

    full_train_df = pd.concat([train_df, val_df], ignore_index=True)

    full_train_df.drop(["timestamp", "title"], axis=1, inplace=True)
    full_train_df = binarize_target(full_train_df)

    cat_cols = [
        "user_id",
        "item_id",
        "gender",
        "genres",
        "age",
        "occupation",
        "zipcode",
    ]

    encoder = LabelEncoder(cat_cols)
    full_train_df_encoded = encoder.fit_transform(full_train_df)

    test_path = (
        Path(DATA_AND_ARTIFACTS_DIR)
        / "train_val_test_splits"
        / "movielens_splits"
        / "test.csv"
    )
    test_df = pd.read_csv(test_path)
    test_df = test_df[full_train_df.columns]
    test_df = binarize_target(test_df)
    test_df_encoded = encoder.transform(test_df)

    # final_features = get_features(gbm="lgb", metric="val_loss")
    # final_cat_cols = [col for col in cat_cols if col in final_features]

    # full_train_df = full_train_df_encoded[final_features + ["rating"]]
    # test_df_encoded = test_df_encoded[final_features + ["rating"]]

    X_train = full_train_df_encoded.drop(columns=["rating"])
    y_train = full_train_df_encoded["rating"]

    X_test = test_df_encoded.drop(columns=["rating"])
    y_test = test_df_encoded["rating"]

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


def train_ctb_model(k_values: List[int] = [5, 10, 20]) -> Dict[int, Dict[str, float]]:

    results_dir = (
        Path(DATA_AND_ARTIFACTS_DIR) / "results" / "ctb_best_setup_ranking_metrics"
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    train_data, test_data = set_ctb_datasets()

    model = ctb.train(
        pool=train_data,
        params={
            "loss_function": "Logloss",
            "eval_metric": "Logloss",
            "iterations": 500,
            "early_stopping_rounds": 50,
            "allow_writing_files": False,
            "verbose": True,
        },
        eval_set=test_data,
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


def train_lgb_model(k_values: List[int] = [5, 10, 20]) -> Dict[int, Dict[str, float]]:

    results_dir = (
        Path(DATA_AND_ARTIFACTS_DIR) / "results" / "lgb_best_setup_ranking_metrics"
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    train_data, test_data = set_lgb_datasets()
    model = lgb.train(
        train_set=train_data,
        params={
            "num_iterations": 1000,
            "objective": "binary",
            "metric": "auc",
        },
        valid_sets=[test_data],
        callbacks=[lgb.early_stopping(stopping_rounds=50)],
    )

    y_pred = model.predict(test_data.data)
    y_test = test_data.get_label()

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
