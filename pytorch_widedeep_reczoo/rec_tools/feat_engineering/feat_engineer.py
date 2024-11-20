import os
from typing import Literal

import pandas as pd

from rec_tools.constants import DATA_AND_ARTIFACTS_DIR
from rec_tools.feat_engineering.utils import (
    save_objects,
    load_movielens,
    load_movie_metadata,
    load_movielens_train_val,
)
from rec_tools.feat_engineering.item_static_feats import process_movie_features
from rec_tools.feat_engineering.item_dynamic_feats import ItemDynamicFeatures
from rec_tools.feat_engineering.user_dynamic_feats import UserDynamicFeatures
from rec_tools.feat_engineering.overview_embeddings import (
    OverviewEmbedder,
    EmbeddingDimensionalityReducer,
)


def run_item_static_feat_engineering(debug: bool = False):
    """Process and engineer static features for movie items in the dataset.

    This function:
    1. Loads MovieLens and movie metadata
    2. Processes movie features with and without LLM
    3. Merges movie overviews from both approaches
    4. Generates embeddings using different methods (Sentence Transformers and Cohere)
    5. Reduces dimensionality of the embeddings

    Args:
        debug (bool, optional): If True, uses a smaller sample of data for testing. Defaults to False.
    """

    movielens_df = load_movielens()
    if debug:
        movielens_df = movielens_df.sample(500, random_state=1).reset_index(drop=True)

    metadata_df = load_movie_metadata()

    # Process movie features (dataset is small so we will run both methods)
    movies_with_overview = process_movie_features(
        movielens_df,
        metadata_df,
        use_llm=False,
    )
    movies_with_overview_llm = process_movie_features(
        movielens_df,
        metadata_df,
        use_llm=True,
    )

    # Merge overviews from both datasets
    movies_with_overview_llm = _merge_movie_overviews(
        movies_with_overview, movies_with_overview_llm
    )

    # Embed
    st_overview_embedder = OverviewEmbedder(replace=True)
    st_overview_embedder.embed_overviews(movies_with_overview_llm)

    ch_overview_embedder = OverviewEmbedder(method="cohere", replace=True)
    ch_overview_embedder.embed_overviews(movies_with_overview_llm)

    # Reduce dimensionality
    st_reducer = EmbeddingDimensionalityReducer(n_components=5, save_suffix="st")
    ch_reducer = EmbeddingDimensionalityReducer(n_components=10, save_suffix="ch")

    _ = st_reducer.reduce(collection=st_overview_embedder.collection)
    _ = ch_reducer.reduce(collection=ch_overview_embedder.collection)


def run_item_dynamic_feat_engineering(prefix: Literal["lpi", "li", "ts"]):
    """Generate dynamic features for items based on historical interactions.

    Args:
        prefix (Literal["lpi", "li", "ts"]): Dataset split prefix:
            - 'lpi': Last Positive Interaction
            - 'li': Last Interaction
            - 'ts': Temporal split
    """

    dirname = f"{prefix}_movielens_splits"

    train, _ = load_movielens_train_val(dirname, "both")

    item_dynamic_feats = ItemDynamicFeatures()
    train_idf = item_dynamic_feats.compute_features(train)

    save_objects(
        [train_idf],
        [f"{prefix}_train_idf.csv"],
        os.path.join(DATA_AND_ARTIFACTS_DIR, "feature_store"),
    )


def run_user_dynamic_feat_engineering(prefix: Literal["lpi", "li", "ts"]):
    """Generate dynamic features for users based on their interaction history.

    Args:
        prefix (Literal["lpi", "li", "ts"]): Dataset split prefix:
            - 'lpi': Last Positive Interaction
            - 'li': Last Interaction
            - 'ts': Temporal split
    """

    dirname = f"{prefix}_movielens_splits"

    train, _ = load_movielens_train_val(dirname, "both")

    user_dynamic_feats = UserDynamicFeatures()
    train_udf = user_dynamic_feats.compute_features(train)

    save_objects(
        [train_udf],
        [f"{prefix}_train_udf.csv"],
        os.path.join(DATA_AND_ARTIFACTS_DIR, "feature_store"),
    )


def _impute_runtime(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing runtime values in the movie dataset.

    First fills NaN values with 0, then replaces 0 values with the median runtime
    of non-zero entries.

    Args:
        df (pd.DataFrame):
            DataFrame containing movie information with 'runtime' column

    Returns:
        pd.DataFrame:
            DataFrame with imputed runtime values
    """
    df = df.copy()
    median_runtime = df[df["runtime"] > 0]["runtime"].median()
    df["runtime"] = df["runtime"].fillna(0.0)
    df.loc[df["runtime"] == 0.0, "runtime"] = median_runtime
    return df


def _merge_movie_overviews(
    movies_with_overview: pd.DataFrame, movies_with_overview_llm: pd.DataFrame
) -> pd.DataFrame:
    """Merge movie overviews from two different sources, prioritizing LLM-generated content.

    This function combines movie overviews from standard processing and LLM-processing,
    using the standard processing as a fallback when LLM-generated overviews are missing.

    Args:
        movies_with_overview (pd.DataFrame):
            DataFrame with standard-processed movie overviews
        movies_with_overview_llm (pd.DataFrame):
            DataFrame with LLM-processed movie overviews

    Returns:
        pd.DataFrame:
            Merged DataFrame with combined overviews and imputed runtime values
    """
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

    merged_df = _impute_runtime(merged_df)

    # replace overview nans with "overview not found"
    merged_df["overview"] = merged_df["overview"].fillna("overview not found")

    return merged_df


if __name__ == "__main__":

    # run_item_static_feat_engineering()

    for prefix in ["lpi", "li", "ts"]:
        print(f"Processing {prefix} dataset")
        run_item_dynamic_feat_engineering(prefix)  # type: ignore
        # run_user_dynamic_feat_engineering(prefix)  # type: ignore
