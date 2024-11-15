from typing import List, Tuple, Literal
from pathlib import Path

import pandas as pd
from pytorch_widedeep.utils import LabelEncoder


def load_and_merge_features(
    use_umap: Literal["st", "ch"] = "st"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    split_path = Path("train_val_test_splits") / "ts_movielens_splits"

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

    ts_train_idf = pd.read_csv("feature_store/ts_train_idf.csv")
    ts_train_udf = pd.read_csv("feature_store/ts_train_udf.csv")
    movie_features = pd.read_csv("feature_store/processed_movie_features_llm.csv")
    movie_features = movie_features[["item_id", "genres", "year", "runtime"]]
    movie_features["runtime"] = movie_features["runtime"].replace(
        0, movie_features["runtime"].median()
    )
    if use_umap == "st":
        umap_results = pd.read_csv("feature_store/umap_results_st.csv")
    elif use_umap == "ch":
        umap_results = pd.read_csv("feature_store/umap_results_ch.csv")
    else:
        raise ValueError("Invalid value for use_umap. Use 'st' or 'ch'.")

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


def binarize_target(df: pd.DataFrame) -> pd.DataFrame:
    df["rating"] = (df["rating"] >= 4).astype(int)
    return df


def prepare_experiment(
    use_umap: Literal["st", "ch"] = "st",
    gbm: Literal["lgbm", "catboost"] = "lgbm",
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], LabelEncoder | None]:
    train_df, val_df = load_and_merge_features(use_umap)

    train_df = binarize_target(train_df)
    val_df = binarize_target(val_df)

    categorical_cols = find_categorical_cols(train_df)
    if gbm == "lgbm":
        encoder = LabelEncoder(columns_to_encode=categorical_cols)
        train_encoded = encoder.fit_transform(train_df)
        val_encoded = encoder.transform(val_df)
        return train_encoded, val_encoded, categorical_cols, encoder
    elif gbm == "catboost":
        return train_df, val_df, categorical_cols, None


if __name__ == "__main__":

    train, val, categorical_cols, encoder = prepare_experiment(gbm="lgbm")
    train, val, categorical_cols, _ = prepare_experiment(gbm="lgbm")
