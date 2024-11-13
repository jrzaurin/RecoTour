import re
import multiprocessing
from typing import Dict, List, Tuple, Union
from functools import partial
from multiprocessing import Pool

import nltk
import pandas as pd
from tqdm import tqdm
from nltk.corpus import stopwords

multiprocessing.set_start_method("fork", force=True)


try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")


def extract_year_and_title(df: pd.DataFrame, title_col: str) -> pd.DataFrame:
    df = df.copy()
    df["year"] = df[title_col].str.extract(r"\((\d{4})\)")
    df[title_col] = df[title_col].str.replace(r"\s*\(\d{4}\)\s*$", "", regex=True)
    return df


def reorder_title(df: pd.DataFrame, title_col: str) -> pd.DataFrame:
    df = df.copy()
    articles = ["The", "A", "An"]

    def reorder_single_title(title: str) -> str:
        for article in articles:
            pattern = f", {article}$"
            if re.search(pattern, title, re.IGNORECASE):
                base_title = re.sub(pattern, "", title, flags=re.IGNORECASE)
                return f"{article} {base_title}"
        return title

    df[title_col] = df[title_col].apply(reorder_single_title)
    return df


class TitleMatcher:

    def match_titles(
        self,
        movielens_df: pd.DataFrame,
        metadata_df: pd.DataFrame,
        match_threshold: float = 0.5,
    ) -> Tuple[Dict[str, Dict[str, Union[str, float]]], List[str]]:
        ml_titles = movielens_df["title"].unique().tolist()

        metadata_df = metadata_df[  # type: ignore
            metadata_df["title"].notnull()
        ].reset_index(drop=True)
        md_titles = metadata_df["title"].tolist()

        matches, unmatched = self._full_match_search(ml_titles, md_titles)

        if unmatched:
            token_match_score = self._token_based_match_search(
                unmatched, metadata_df["title"].tolist()
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

        return matches, token_unmatched

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

    def _token_match_score(self, title1: str, title2: str) -> float:
        tokens1 = set(self._clean_and_tokenize(title1))
        tokens2 = set(self._clean_and_tokenize(title2))

        if not tokens1 or not tokens2:
            return 0.0

        matches = len(tokens1.intersection(tokens2))
        total_unique = len(tokens1.union(tokens2))

        return matches / total_unique

    @staticmethod
    def _clean_and_tokenize(text: str) -> List[str]:
        text = text.lower()
        text = re.sub(r"[^a-za-z0-9\s]", "", text)
        words = text.split()
        stop_words = set(stopwords.words("english"))
        return [w for w in words if w not in stop_words and len(w) >= 3]

    @staticmethod
    def _reorder_title_article(title: str) -> str:
        articles = ["The", "A", "An"]

        for article in articles:
            pattern = f", {article}$"
            if re.search(pattern, title, re.IGNORECASE):
                base_title = re.sub(pattern, "", title, flags=re.IGNORECASE)
                return f"{article} {base_title}"

        return title
