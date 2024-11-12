from typing import Literal

import pandas as pd

from rec_zoo.feat_engineering.utils import (
    save_objects,
    load_movielens,
    load_movie_metadata,
    load_movielens_train_val,
)
from rec_zoo.feat_engineering.item_static_feats import process_movie_features
from rec_zoo.feat_engineering.item_dynamic_feats import ItemDynamicFeatures
from rec_zoo.feat_engineering.user_dynamic_feats import UserDynamicFeatures
from rec_zoo.feat_engineering.overview_embeddings import (
    OverviewEmbedder,
    EmbeddingDimensionalityReducer,
)


def _merge_movie_overviews(
    movies_with_overview: pd.DataFrame, movies_with_overview_llm: pd.DataFrame
) -> pd.DataFrame:
    """
    Fill empty overviews in the LLM dataset with non-empty overviews from the regular dataset

    Args:
        movies_with_overview (pd.DataFrame): Original movie dataset with overviews
        movies_with_overview_llm (pd.DataFrame): Movie dataset with LLM-generated overviews

    Returns:
        pd.DataFrame: Updated LLM dataset with filled overviews
    """
    # Create a copy to avoid modifying the original
    merged_df = movies_with_overview_llm.copy()

    # Find rows where overview is empty in LLM dataset
    empty_overviews = merged_df["overview"] == ""

    # For each empty overview, try to fill it from the regular dataset
    for idx in merged_df[empty_overviews].index:
        movie_id = merged_df.loc[idx, "item_id"]
        original_overview = movies_with_overview.loc[
            movies_with_overview["item_id"] == movie_id, "overview"
        ].iloc[0]

        if original_overview != "":
            merged_df.loc[idx, "overview"] = original_overview

    return merged_df


def run_item_static_feat_engineering():

    movielens_df = load_movielens()
    metadata_df = load_movie_metadata()

    # Process movie features (dataset is small so we will run both methods)
    movies_with_overview = process_movie_features(movielens_df, metadata_df)
    movies_with_overview_llm = process_movie_features(
        movielens_df, metadata_df, use_llm=True
    )

    # Merge overviews from both datasets
    movies_with_overview_llm = _merge_movie_overviews(
        movies_with_overview, movies_with_overview_llm
    )

    # Embed
    st_overview_embedder = OverviewEmbedder()
    st_overview_embedder.embed_overviews(movies_with_overview_llm)
    save_objects([st_overview_embedder], ["st_overview_embedder"], "artifacts")

    ch_overview_embedder = OverviewEmbedder(method="cohere")
    ch_overview_embedder.embed_overviews(movies_with_overview_llm)
    save_objects([ch_overview_embedder], ["ch_overview_embedder"], "artifacts")

    # Reduce dimensionality
    reducer = EmbeddingDimensionalityReducer()

    _ = reducer.reduce(collection=st_overview_embedder.collection)
    _ = reducer.reduce(collection=ch_overview_embedder.collection)


def run_item_dynamic_feat_engineering(prefix: Literal["li", "ts"]):

    dirname = f"{prefix}_movielens_splits"

    train, val = load_movielens_train_val(dirname, "both")

    item_dynamic_feats = ItemDynamicFeatures()
    train_idf = item_dynamic_feats.compute_features(train)
    va_idf = pd.merge(
        val[["item_id"]].drop_duplicates(), train_idf, on="item_id", how="left"
    )
    save_objects(
        [train_idf, va_idf],
        [f"{prefix}_train_idf", f"{prefix}_va_idf"],
        "feature_store",
    )


def run_user_dynamic_feat_engineering(prefix: Literal["li", "ts"]):

    dirname = f"{prefix}_movielens_splits"

    train, val = load_movielens_train_val(dirname, "both")

    user_dynamic_feats = UserDynamicFeatures()
    train_udf = user_dynamic_feats.compute_features(train)
    va_udf = pd.merge(
        val[["user_id"]].drop_duplicates(), train_udf, on="user_id", how="left"
    )
    save_objects(
        [train_udf, va_udf],
        [f"{prefix}_train_udf", f"{prefix}_va_udf"],
        "feature_store",
    )
