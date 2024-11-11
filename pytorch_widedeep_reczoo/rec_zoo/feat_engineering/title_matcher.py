import re
import pickle
import asyncio
import multiprocessing
from typing import Dict, List, Tuple, Union
from pathlib import Path
from functools import partial
from multiprocessing import Pool

import nltk
import pandas as pd
from tqdm import tqdm
from nltk.corpus import stopwords

from rec_zoo.tokens_and_api_keys import OPENAI_API_KEY
from rec_zoo.feat_engineering.openai_get_title import process_movies_concurrent

multiprocessing.set_start_method("fork", force=True)


try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")


def load_movie_lens(full_path: str | None = None) -> pd.DataFrame:
    ml_path = (
        Path(full_path)
        if full_path
        else Path("data/ml-1m/movielens_ratings_with_info.csv")
    )
    return pd.read_csv(ml_path)


def load_movie_metadata(full_path: str | None = None) -> pd.DataFrame:
    ml_path = (
        Path(full_path) if full_path else Path("data/ml-1m/movies_metadata.csv.zip")
    )
    return pd.read_csv(ml_path)


async def get_llm_matches(titles: List[str]) -> Dict[str, Dict[str, str]]:
    llm_results = await process_movies_concurrent(
        titles, OPENAI_API_KEY, max_concurrent=5
    )
    return dict(zip(titles, llm_results))


class TitleMatcher:

    def match_titles(
        self,
        movielens_df: pd.DataFrame,
        metadata_df: pd.DataFrame,
        movielens_title_col: str = "title",
        metadata_title_col: str = "title",
        match_threshold: float = 0.5,
    ) -> Tuple[Dict[str, Dict[str, Union[str, float]]], List[str]]:
        ml_df = self._extract_year_and_title(movielens_df, movielens_title_col)

        titles = ml_df["clean_title"].unique().tolist()
        ml_titles = [self._reorder_title_article(title) for title in titles]

        metadata_df = metadata_df[
            metadata_df[metadata_title_col].notnull()
        ].reset_index(drop=True)

        md_titles = metadata_df[metadata_title_col].tolist()

        matches, unmatched = self._full_match_search(ml_titles, md_titles)

        if unmatched:
            token_match_score = self._token_based_match_search(
                unmatched, metadata_df[metadata_title_col].tolist()
            )
            token_matches = {
                k: v for k, v in token_match_score.items() if v["score"] >= match_threshold  # type: ignore[operator]
            }
            token_unmatched = [
                k for k, v in token_match_score.items() if v["score"] < match_threshold  # type: ignore[operator]
            ]
            matches.update(token_matches)
        else:
            token_unmatched = []

        # save the two objects to the local directory (pickel)
        with open("matches.pkl", "wb") as m:
            pickle.dump(matches, m)

        with open("token_unmatched.pkl", "wb") as tu:
            pickle.dump(token_unmatched, tu)

        return matches, token_unmatched

    def get_movie_info_from_matches(
        self,
        matches: Dict[str, Dict[str, Union[str, float]]],
        metadata_df: pd.DataFrame,
        metadata_title_col: str = "title",
    ) -> Dict[str, Dict[str, str]]:
        movie_info = {}
        for ml_title, match_info in matches.items():
            matched_title = match_info["match"]
            movie_data = metadata_df[metadata_df[metadata_title_col] == matched_title]
            if not movie_data.empty:
                movie_info[ml_title] = {
                    "overview": movie_data["overview"].iloc[0],
                    "runtime": str(movie_data["runtime"].iloc[0]),
                }
        return movie_info

    def create_movie_info_df(
        self,
        matches_info: Dict[str, Dict[str, str]],
        llm_matches: Dict[str, Dict[str, str]] | None = None,
    ) -> pd.DataFrame:
        # Combine dictionaries if llm_matches is provided
        all_info = matches_info.copy()
        if llm_matches:
            all_info.update(llm_matches)

        # Create DataFrame
        df = pd.DataFrame(
            [{"title": title, **info} for title, info in all_info.items()]
        )

        return df[["title", "overview", "runtime"]]

    @staticmethod
    def _clean_and_tokenize(text: str) -> List[str]:
        text = text.lower()
        text = re.sub(r"[^a-za-z0-9\s]", "", text)
        words = text.split()
        stop_words = set(stopwords.words("english"))
        return [w for w in words if w not in stop_words and len(w) >= 3]

    def _token_match_score(self, title1: str, title2: str) -> float:
        tokens1 = set(self._clean_and_tokenize(title1))
        tokens2 = set(self._clean_and_tokenize(title2))

        if not tokens1 or not tokens2:
            return 0.0

        matches = len(tokens1.intersection(tokens2))
        total_unique = len(tokens1.union(tokens2))

        return matches / total_unique

    def _full_match_search(
        self, input_titles: List[str], lookup_titles: List[str]
    ) -> Tuple[Dict[str, Dict[str, str | float]], List[str]]:
        lookup_set: Dict[str, str] = {title.lower(): title for title in lookup_titles}

        matches = {}
        unmatched = []

        for title in input_titles:
            lower_title = title.lower()
            if lower_title in lookup_set:
                matches[title] = {"match": lookup_set[lower_title], "score": 1.0}
            else:
                unmatched.append(title)

        return matches, unmatched  # type: ignore[return-value]

    def _match_single_title(
        self, title: str, sorted_lookup: List[str]
    ) -> Dict[str, Union[str, float]]:
        tokens = self._clean_and_tokenize(title)
        if not tokens:
            return {"match": "", "score": 0.0}

        start_letter = min(token[0] for token in tokens)

        start_idx = 0
        for idx, lookup_title in enumerate(sorted_lookup):
            if lookup_title.lower().startswith(start_letter):
                start_idx = idx
                break

        best_match = ""
        best_score = 0.0

        for lookup_title in sorted_lookup[start_idx:]:
            score = self._token_match_score(title, lookup_title)
            if score > best_score:
                best_score = score
                best_match = lookup_title

                if score == 1.0:
                    break

        return {"match": best_match, "score": best_score}

    def _token_based_match_search(
        self, input_titles: List[str], lookup_titles: List[str]
    ) -> Dict[str, Dict[str, str | float]]:
        sorted_lookup = sorted(lookup_titles)
        match_func = partial(self._match_single_title, sorted_lookup=sorted_lookup)

        with Pool() as pool:
            results = list(
                tqdm(
                    pool.imap(match_func, input_titles),
                    total=len(input_titles),
                    desc="Matching titles",
                )
            )

        return dict(zip(input_titles, results))

    @staticmethod
    def _reorder_title_article(title: str) -> str:
        articles = ["The", "A", "An"]

        for article in articles:
            pattern = f", {article}$"
            if re.search(pattern, title, re.IGNORECASE):
                base_title = re.sub(pattern, "", title, flags=re.IGNORECASE)
                return f"{article} {base_title}"

        return title

    def _extract_year_and_title(self, df: pd.DataFrame, title_col: str) -> pd.DataFrame:
        df = df.copy()
        df["year"] = df[title_col].str.extract(r"\((\d{4})\)")
        df["clean_title"] = df[title_col].str.replace(
            r"\s*\(\d{4}\)\s*$", "", regex=True
        )
        return df


if __name__ == "__main__":

    tm = TitleMatcher()
    ml_df = load_movie_lens()
    md_df = load_movie_metadata()

    matches, token_unmatched = tm.match_titles(ml_df, md_df)

    matches_info = tm.get_movie_info_from_matches(matches, md_df)

    llm_matches = asyncio.run(get_llm_matches(token_unmatched[:10]))

    final_df = tm.create_movie_info_df(matches_info, llm_matches)
