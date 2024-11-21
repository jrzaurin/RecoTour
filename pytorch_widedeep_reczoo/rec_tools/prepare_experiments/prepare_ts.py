from typing import List, Tuple, Literal
from pathlib import Path

import pandas as pd
from pytorch_widedeep.utils import LabelEncoder

from rec_tools.constants import DATA_AND_ARTIFACTS_DIR


def load_splits() -> Tuple[pd.DataFrame, pd.DataFrame]:
    split_path = (
        Path(DATA_AND_ARTIFACTS_DIR) / "train_val_test_splits" / "ts_movielens_splits"
    )

    train_df = pd.read_csv(split_path / "train.csv")
    val_df = pd.read_csv(split_path / "val.csv")

    return train_df, val_df


def load_and_merge_features_with_raw_text() -> Tuple[pd.DataFrame, pd.DataFrame]:
    split_path = (
        Path(DATA_AND_ARTIFACTS_DIR) / "train_val_test_splits" / "ts_movielens_splits"
    )
    movie_features_path = (
        Path(DATA_AND_ARTIFACTS_DIR) / "feature_store/processed_movie_features_llm.csv"
    )

    cols_to_keep = [
        "user_id",
        "item_id",
        "rating",
        "gender",
        "age",
        "occupation",
        "zipcode",
    ]
    train_df = pd.read_csv(split_path / "train.csv")[cols_to_keep]
    val_df = pd.read_csv(split_path / "val.csv")[cols_to_keep]

    movie_features = pd.read_csv(movie_features_path)
    movie_features = movie_features[
        ["item_id", "genres", "year", "overview", "runtime"]
    ]
    movie_features["runtime"] = movie_features["runtime"].replace(
        0, movie_features["runtime"].median()
    )

    train_df = train_df.merge(movie_features, on="item_id", how="left")
    val_df = val_df.merge(movie_features, on="item_id", how="left")

    return train_df, val_df


def load_and_merge_all_features(
    use_umap: Literal["st", "ch"] = "st"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    split_path = (
        Path(DATA_AND_ARTIFACTS_DIR) / "train_val_test_splits" / "ts_movielens_splits"
    )
    movie_features_path = (
        Path(DATA_AND_ARTIFACTS_DIR) / "feature_store/processed_movie_features_llm.csv"
    )
    item_dynamic_features_path = (
        Path(DATA_AND_ARTIFACTS_DIR) / "feature_store" / "ts_train_idf.csv"
    )
    user_dynamic_features_path = (
        Path(DATA_AND_ARTIFACTS_DIR) / "feature_store" / "ts_train_udf.csv"
    )
    umap_results_path = (
        Path(DATA_AND_ARTIFACTS_DIR) / "feature_store" / f"umap_results_{use_umap}.csv"
    )

    cols_to_keep = [
        "user_id",
        "item_id",
        "rating",
        "gender",
        "age",
        "occupation",
        "zipcode",
    ]

    train_df = pd.read_csv(split_path / "train.csv")[cols_to_keep]
    val_df = pd.read_csv(split_path / "val.csv")[cols_to_keep]

    ts_train_idf = pd.read_csv(item_dynamic_features_path)
    ts_train_udf = pd.read_csv(user_dynamic_features_path)

    movie_features = pd.read_csv(movie_features_path)
    movie_features = movie_features[["item_id", "genres", "year", "runtime"]]
    movie_features["runtime"] = movie_features["runtime"].replace(
        0, movie_features["runtime"].median()
    )
    umap_results = pd.read_csv(umap_results_path)

    train_df = train_df.merge(movie_features, on="item_id", how="left")
    val_df = val_df.merge(movie_features, on="item_id", how="left")

    train_df = train_df.merge(ts_train_idf, on="item_id", how="left")
    val_df = val_df.merge(ts_train_idf, on="item_id", how="left")

    train_df = train_df.merge(
        ts_train_udf, on="user_id", how="left", suffixes=("_item", "_user")
    )
    val_df = val_df.merge(
        ts_train_udf, on="user_id", how="left", suffixes=("_item", "_user")
    )

    train_df = train_df.merge(umap_results, on="item_id", how="left")
    val_df = val_df.merge(umap_results, on="item_id", how="left")

    return train_df, val_df


def find_categorical_cols(train_df: pd.DataFrame) -> list[str]:
    categorical_cols = ["user_id", "item_id"]
    for col in train_df.columns:
        if col == "rating":
            continue
        is_numeric = train_df[col].dtype in ["int32", "int64", "float32", "float64"]
        n_unique = train_df[col].nunique()
        if (train_df[col].dtype == "object") or (is_numeric and n_unique < 250):
            categorical_cols.append(col)
    return categorical_cols


def impute_float_categorical_cols(
    df: pd.DataFrame, categorical_cols: List[str]
) -> pd.DataFrame:
    # first, if there are any float cols, replace nans with -1 and cast to int
    float_categorical_cols = [
        col for col in categorical_cols if df[col].dtype == "float64"
    ]
    df[float_categorical_cols] = df[float_categorical_cols].fillna(-1).astype(int)

    # finally, if there is any cat col left, if is object, replace with
    # "nan_cat", if is int, replace with -1
    for col in categorical_cols:
        if df[col].dtype == "object":
            df[col] = df[col].fillna("nan_cat")
        elif df[col].dtype == "int64":
            df[col] = df[col].fillna(-1)

    return df


def binarize_target(df: pd.DataFrame) -> pd.DataFrame:
    df["rating"] = (df["rating"] >= 4).astype(int)
    return df


def prepare_experiment_without_feat_engineering(
    gbm: Literal["lgbm", "catboost"] = "lgbm",
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], LabelEncoder | None]:

    train_df, val_df = load_splits()

    for df in [train_df, val_df]:
        df.drop(["timestamp", "title"], axis=1, inplace=True)
        df = binarize_target(df)

    cat_cols = [
        "user_id",
        "item_id",
        "gender",
        "genres",
        "age",
        "occupation",
        "zipcode",
    ]

    if gbm == "lgbm":
        encoder = LabelEncoder(cat_cols)
        train_encoded = encoder.fit_transform(train_df)
        val_encoded = encoder.transform(val_df)
        return train_encoded, val_encoded, cat_cols, encoder
    else:
        return train_df, val_df, cat_cols, None


def prepare_experiment_with_feature_engineering(
    use_umap: Literal["st", "ch"] = "st",
    gbm: Literal["lgbm", "catboost"] = "lgbm",
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], LabelEncoder | None]:
    train_df, val_df = load_and_merge_all_features(use_umap)

    train_df = binarize_target(train_df)
    val_df = binarize_target(val_df)

    categorical_cols = find_categorical_cols(train_df)
    train_df = impute_float_categorical_cols(train_df, categorical_cols)
    val_df = impute_float_categorical_cols(val_df, categorical_cols)

    if gbm == "lgbm":
        encoder = LabelEncoder(columns_to_encode=categorical_cols)
        train_encoded = encoder.fit_transform(train_df)
        val_encoded = encoder.transform(val_df)
        return train_encoded, val_encoded, categorical_cols, encoder
    elif gbm == "catboost":
        return train_df, val_df, categorical_cols, None


def prepare_experiment_for_catboost_with_text() -> (
    Tuple[pd.DataFrame, pd.DataFrame, List[str]]
):
    train_df, val_df = load_and_merge_features_with_raw_text()
    categorical_cols = find_categorical_cols(train_df)
    categorical_cols = [
        col for col in categorical_cols if col not in ["overview", "runtime"]
    ]
    return train_df, val_df, categorical_cols


if __name__ == "__main__":

    train, val, categorical_cols, encoder = prepare_experiment_with_feature_engineering(
        gbm="lgbm"
    )
    train, val, categorical_cols, _ = prepare_experiment_with_feature_engineering(
        gbm="catboost"
    )
