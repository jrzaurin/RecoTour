from typing import List, Tuple, Literal
from pathlib import Path

import pandas as pd

from rec_tools.constants import (
    DATA_DIR,
    FEATURE_STORE_DIR,
    MOVIELENS_SPLITS_DIR,
    TRAIN_VAL_TEST_SPLITS_DIR,
    TEMPORAL_MOVIELENS_SPLIT_DIR,
)


def binarize_target(df: pd.DataFrame) -> pd.DataFrame:
    df["rating"] = (df["rating"] >= 4).astype(int)
    return df


def load_and_merge_features(
    split: Literal["train_val", "test"] = "train_val",
    use_umap: Literal["st", "ch"] = "st",
) -> pd.DataFrame | Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and merge features for train/val or test data.

    Args:
        split: Which split to load ("train", "val", or "test")
        use_umap: Which UMAP features to use ("st" or "ch")

    Returns:
        Single DataFrame for test split, or tuple of (train_df, val_df) for train/val
    """
    cols_to_keep = [
        "user_id",
        "item_id",
        "rating",
        "gender",
        "age",
        "occupation",
        "zipcode",
    ]

    if split == "test":
        split_path = Path(DATA_DIR) / TRAIN_VAL_TEST_SPLITS_DIR / MOVIELENS_SPLITS_DIR
        df = pd.read_csv(split_path / "test.csv")[cols_to_keep]
    else:
        split_path = (
            Path(DATA_DIR) / TRAIN_VAL_TEST_SPLITS_DIR / TEMPORAL_MOVIELENS_SPLIT_DIR
        )
        df = pd.read_csv(split_path / "train.csv")[cols_to_keep]
        val_df = pd.read_csv(split_path / "val.csv")[cols_to_keep]

    # Load all feature files
    movie_features_path = (
        Path(DATA_DIR) / FEATURE_STORE_DIR / "processed_movie_features_llm.csv"
    )
    item_dynamic_features_path = Path(DATA_DIR) / FEATURE_STORE_DIR / "ts_train_idf.csv"
    user_dynamic_features_path = Path(DATA_DIR) / FEATURE_STORE_DIR / "ts_train_udf.csv"
    umap_results_path = (
        Path(DATA_DIR) / FEATURE_STORE_DIR / f"umap_results_{use_umap}.csv"
    )

    # Load and prepare features
    movie_features = pd.read_csv(movie_features_path)
    movie_features = movie_features[["item_id", "genres", "year", "runtime"]]
    movie_features["runtime"] = movie_features["runtime"].replace(
        0, movie_features["runtime"].median()
    )
    ts_train_idf = pd.read_csv(item_dynamic_features_path)
    ts_train_udf = pd.read_csv(user_dynamic_features_path)
    umap_results = pd.read_csv(umap_results_path)

    # Merge all features
    df = df.merge(movie_features, on="item_id", how="left")
    df = df.merge(ts_train_idf, on="item_id", how="left")
    df = df.merge(ts_train_udf, on="user_id", how="left", suffixes=("_item", "_user"))
    df = df.merge(umap_results, on="item_id", how="left")

    if split == "train_val":
        val_df = val_df.merge(movie_features, on="item_id", how="left")
        val_df = val_df.merge(ts_train_idf, on="item_id", how="left")
        val_df = val_df.merge(
            ts_train_udf, on="user_id", how="left", suffixes=("_item", "_user")
        )
        val_df = val_df.merge(umap_results, on="item_id", how="left")
        return df, val_df

    return df


def find_categorical_cols(
    train_df: pd.DataFrame, cat_threshold: int = 250
) -> list[str]:
    # Find categorical cols based on threshold and dtype. 250 is arbitrary.
    categorical_cols = ["user_id", "item_id"]
    for col in train_df.columns:
        if col == "rating":
            continue
        is_numeric = train_df[col].dtype in ["int32", "int64", "float32", "float64"]
        n_unique = train_df[col].nunique()
        if (train_df[col].dtype == "object") or (
            is_numeric and n_unique < cat_threshold
        ):
            categorical_cols.append(col)
    return categorical_cols


def impute_categorical_cols(
    df: pd.DataFrame, categorical_cols: List[str]
) -> pd.DataFrame:
    # This could be done better. But for now I just want to run some quick
    # experiments. In general, NaNs will be treated as a new category.

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


def experiment_without_feat_engineering() -> (
    Tuple[pd.DataFrame, pd.DataFrame, List[str]]
):

    split_path = (
        Path(DATA_DIR) / TRAIN_VAL_TEST_SPLITS_DIR / TEMPORAL_MOVIELENS_SPLIT_DIR
    )
    train_df = pd.read_csv(split_path / "train.csv")
    val_df = pd.read_csv(split_path / "val.csv")

    # hardcoded cat cols. They are all categorical.
    cat_cols = [
        "user_id",
        "item_id",
        "gender",
        "genres",
        "age",
        "occupation",
        "zipcode",
    ]
    train_df = train_df[cat_cols + ["rating"]]
    val_df = val_df[cat_cols + ["rating"]]

    train_df = binarize_target(train_df)
    val_df = binarize_target(val_df)

    return train_df, val_df, cat_cols


def experiment_with_feature_engineering(
    use_umap: Literal["st", "ch"] = "st",
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    train_df, val_df = load_and_merge_features(split="train_val", use_umap=use_umap)

    train_df = binarize_target(train_df)
    val_df = binarize_target(val_df)

    cat_cols = find_categorical_cols(train_df)

    train_df = impute_categorical_cols(train_df, cat_cols)
    val_df = impute_categorical_cols(val_df, cat_cols)

    return train_df, val_df, cat_cols


def experiment_for_catboost_with_text() -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    split_path = (
        Path(DATA_DIR) / TRAIN_VAL_TEST_SPLITS_DIR / TEMPORAL_MOVIELENS_SPLIT_DIR
    )
    train_df = pd.read_csv(split_path / "train.csv")
    val_df = pd.read_csv(split_path / "val.csv")

    # hardcoded cat cols. They are all categorical.
    cat_cols = [
        "user_id",
        "item_id",
        "gender",
        "genres",
        "age",
        "occupation",
        "zipcode",
    ]
    train_df = train_df[cat_cols + ["rating"]]
    val_df = val_df[cat_cols + ["rating"]]

    train_df = binarize_target(train_df)
    val_df = binarize_target(val_df)

    # Load all feature files
    movie_features_path = (
        Path(DATA_DIR) / FEATURE_STORE_DIR / "processed_movie_features_llm.csv"
    )
    movie_features = pd.read_csv(movie_features_path)
    movie_features = movie_features[["item_id", "overview"]]

    train_df = train_df.merge(movie_features, on="item_id", how="left")
    val_df = val_df.merge(movie_features, on="item_id", how="left")

    return train_df, val_df, cat_cols
