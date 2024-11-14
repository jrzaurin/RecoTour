from typing import Literal

import fire
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
    merged_df = movies_with_overview_llm.copy()

    empty_overviews = merged_df["overview"] == "overview not found"

    for idx in merged_df[empty_overviews].index:
        movie_id = merged_df.loc[idx, "item_id"]
        matching_rows = movies_with_overview[
            movies_with_overview["item_id"] == movie_id
        ]

        if not matching_rows.empty:
            merged_df.loc[idx, ["overview", "runtime"]] = matching_rows[
                ["overview", "runtime"]
            ].iloc[0]

    missing_movies = movies_with_overview[
        ~movies_with_overview["item_id"].isin(merged_df["item_id"])
    ]

    if not missing_movies.empty:
        merged_df = pd.concat([merged_df, missing_movies], ignore_index=True)

    return merged_df


def run_item_static_feat_engineering(debug: bool = False):

    movielens_df = load_movielens()
    if debug:
        movielens_df = movielens_df.sample(500, random_state=1).reset_index(drop=True)

    metadata_df = load_movie_metadata()

    # Process movie features (dataset is small so we will run both methods)
    movies_with_overview = process_movie_features(
        movielens_df, metadata_df, use_llm=False, save_dir="feature_store"
    )
    movies_with_overview_llm = process_movie_features(
        movielens_df, metadata_df, use_llm=True, save_dir="feature_store"
    )

    # Merge overviews from both datasets
    movies_with_overview_llm = _merge_movie_overviews(
        movies_with_overview, movies_with_overview_llm
    )

    # Embed
    st_overview_embedder = OverviewEmbedder()
    st_overview_embedder.embed_overviews(movies_with_overview_llm)

    ch_overview_embedder = OverviewEmbedder(method="cohere")
    ch_overview_embedder.embed_overviews(movies_with_overview_llm)

    # Reduce dimensionality
    st_reducer = EmbeddingDimensionalityReducer(n_components=5, save_suffix="st")
    ch_reducer = EmbeddingDimensionalityReducer(n_components=10, save_suffix="ch")

    _ = st_reducer.reduce(collection=st_overview_embedder.collection)
    _ = ch_reducer.reduce(collection=ch_overview_embedder.collection)


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


if __name__ == "__main__":

    fire.Fire(run_item_static_feat_engineering)
