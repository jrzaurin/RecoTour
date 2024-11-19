import asyncio
from typing import Dict, List

import pandas as pd

from rec_tools.constants import DATA_AND_ARTIFACTS_DIR
from rec_tools.tokens_and_api_keys import OPENAI_API_KEY
from rec_tools.feat_engineering.utils import (
    save_objects,
    load_movielens,
    load_movie_metadata,
)
from rec_tools.feat_engineering.title_matcher import (
    TitleMatcher,
    reorder_title,
    extract_year_and_title,
)
from rec_tools.feat_engineering.openai_get_title import process_movies_concurrent


def process_movie_features(
    movies_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    use_llm: bool = False,
    save_dir: str | None = f"{DATA_AND_ARTIFACTS_DIR}/feature_store",
    replace: bool = False,
) -> pd.DataFrame:
    """Process movie features by extracting and enriching movie information.

    This function processes movie features through several steps:
    1. Extracts year and standardizes movie titles
    2. Processes genre information
    3. Retrieves movie information either through metadata matching or via an
    LLM
    4. Combines all information into a final DataFrame

    Args:
        movies_df (pd.DataFrame):
            Input DataFrame containing movie information with at least 'title' column
        metadata_df (pd.DataFrame):
            Reference DataFrame containing movie metadata
        use_llm (bool, optional):
            Whether to use LLM for retrieving movie information. Defaults to False
        save_dir (str | None, optional):
            Directory to save processed results. If None, results won't be saved
        replace (bool, optional):
            Whether to replace existing saved files. Defaults to False

    Returns:
        pd.DataFrame: Processed DataFrame containing enriched movie information with columns:
            - Original columns from movies_df
            - overview: Movie plot overview
            - runtime: Movie duration in minutes

    Note:
        If save_dir is provided and replace=False, the function will first try to load
        existing processed data before performing any processing.
    """
    if save_dir:
        save_fname = (
            "processed_movie_features_llm.csv"
            if use_llm
            else "processed_movie_features.csv"
        )
        full_path = f"{save_dir}/{save_fname}"

        if not replace:
            try:
                return pd.read_csv(full_path)
            except FileNotFoundError:
                pass

    # Step 1: Extract year and title, and re-order title if neccessary
    processed_df = extract_year_and_title(movies_df, "title")
    processed_df = reorder_title(processed_df, "title")

    # Step 2: Process genres
    processed_df = process_genres(processed_df, "genres")

    # Step 3: Get movie information
    matches_info: Dict[str, Dict[str, str | float]] = {}
    llm_info: Dict[str, Dict[str, str | float]] = {}

    if use_llm:
        llm_info = asyncio.run(get_llm_info(processed_df["title"].tolist(), save_dir))
    else:
        matches_info, token_unmatched = get_title_matches(
            processed_df, metadata_df, save_dir
        )
        if token_unmatched:
            llm_info = asyncio.run(get_llm_info(token_unmatched, save_dir))

    # Step 4: Create final DataFrame
    movie_info_df = create_movie_info_df(matches_info, llm_info)

    final_df = processed_df.merge(
        movie_info_df[["title", "overview", "runtime"]], on="title", how="left"
    )

    if save_dir:
        save_objects([final_df], [save_fname], save_dir)

    return final_df


def process_genres(df: pd.DataFrame, genre_col: str) -> pd.DataFrame:
    """Process genre information in the DataFrame.

    Args:
        df (pd.DataFrame):
            Input DataFrame containing genre information
        genre_col (str):
            Name of the column containing genre information

    Returns:
        pd.DataFrame:
            DataFrame with processed genre information (lowercase and '|' replaced with '_')
    """
    df[genre_col] = df[genre_col].str.lower().str.replace("|", "_")
    return df


def get_movie_info_from_matches(
    matches: Dict[str, Dict[str, str | float]],
    metadata_df: pd.DataFrame,
    metadata_title_col: str = "title",
) -> Dict[str, Dict[str, str | float]]:
    """Extract movie information from matched titles using metadata.

    Args:
        matches (Dict[str, Dict[str, str | float]]):
            Dictionary of matched movie titles
        metadata_df (pd.DataFrame):
            DataFrame containing movie metadata
        metadata_title_col (str, optional):
            Name of the title column in metadata_df. Defaults to "title"

    Returns:
        Dict[str, Dict[str, str | float]]:
            Dictionary mapping movie titles to their overview and runtime information
    """
    movie_info = {}
    for ml_title, match_info in matches.items():
        matched_title = match_info["match"]
        movie_data = metadata_df[metadata_df[metadata_title_col] == matched_title]
        if not movie_data.empty:
            movie_info[ml_title] = {
                "overview": movie_data["overview"].iloc[0],
                "runtime": movie_data["runtime"].iloc[0],
            }
    return movie_info


def get_title_matches(
    processed_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    save_dir: str | None = None,
) -> tuple[dict, list[str]]:
    """Match titles and return matches and unmatched titles."""
    title_matcher = TitleMatcher()
    matches, token_unmatched = title_matcher.match_titles(processed_df, metadata_df)
    matches_info = get_movie_info_from_matches(matches, metadata_df)

    if save_dir:
        save_objects(
            [matches, token_unmatched, matches_info],
            ["matches.pkl", "token_unmatched.pkl", "matches_info.csv"],
            save_dir,
        )

    return matches_info, token_unmatched


async def get_movie_info_with_llm(
    titles: List[str],
) -> Dict[str, Dict[str, str | float]]:
    """Retrieve movie information using LLM for a list of titles.

    Args:
        titles (List[str]):
            List of movie titles to process

    Returns:
        Dict[str, Dict[str, str | float]]:
            Dictionary mapping titles to their LLM-generated information
    """
    llm_results = await process_movies_concurrent(
        titles, OPENAI_API_KEY, max_concurrent=5
    )
    return dict(zip(titles, llm_results))


async def get_llm_info(titles: list[str], save_dir: str | None = None) -> dict:
    """Get movie information using LLM."""
    llm_info = await get_movie_info_with_llm(titles)

    if save_dir:
        save_objects([llm_info], ["llm_info.csv"], save_dir)

    return llm_info


def create_movie_info_df(
    matches_info: Dict[str, Dict[str, str | float]] = {},
    llm_matches: Dict[str, Dict[str, str | float]] = {},
) -> pd.DataFrame:
    """Create a DataFrame from matched and LLM-generated movie information.

    Args:
        matches_info (Dict[str, Dict[str, str | float]], optional):
            Dictionary of matched movie information. Defaults to {}
        llm_matches (Dict[str, Dict[str, str | float]], optional):
            Dictionary of LLM-generated movie information. Defaults to {}

    Returns:
        pd.DataFrame:
            DataFrame containing combined movie information with columns: title, overview, and runtime
    """
    all_info = matches_info.copy()
    all_info.update(llm_matches)
    df = pd.DataFrame([{"title": title, **info} for title, info in all_info.items()])
    return df[["title", "overview", "runtime"]]


def manual_update_df_from_dict(df: pd.DataFrame, overview_dict: dict) -> pd.DataFrame:
    """Update dataframe overviews and runtimes from dictionary based on title matches."""
    result_df = df.copy()

    for title, data in overview_dict.items():
        mask = result_df["title"] == title
        if any(mask):
            result_df.loc[mask, "overview"] = data["overview"]
            result_df.loc[mask, "runtime"] = data["runtime"]

    return result_df


if __name__ == "__main__":
    ml_df = load_movielens()
    md_df = load_movie_metadata()

    processed_df = process_movie_features(ml_df, md_df)
