import asyncio
from typing import Dict, List

import pandas as pd

from rec_zoo.tokens_and_api_keys import OPENAI_API_KEY
from rec_zoo.feat_engineering.utils import (
    save_objects,
    load_movielens,
    load_movie_metadata,
)
from rec_zoo.feat_engineering.title_matcher import (
    TitleMatcher,
    reorder_title,
    extract_year_and_title,
)
from rec_zoo.feat_engineering.openai_get_title import process_movies_concurrent


def process_movie_features(
    movies_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    use_llm: bool = False,
    save_dir: str | None = None,
    replace: bool = False,
) -> pd.DataFrame:
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
    df[genre_col] = df[genre_col].str.lower().str.replace("|", "_")
    return df


def get_movie_info_from_matches(
    matches: Dict[str, Dict[str, str | float]],
    metadata_df: pd.DataFrame,
    metadata_title_col: str = "title",
) -> Dict[str, Dict[str, str | float]]:
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
    all_info = matches_info.copy()
    all_info.update(llm_matches)
    df = pd.DataFrame([{"title": title, **info} for title, info in all_info.items()])
    return df[["title", "overview", "runtime"]]


if __name__ == "__main__":
    ml_df = load_movielens()
    md_df = load_movie_metadata()

    processed_df = process_movie_features(ml_df, md_df)
