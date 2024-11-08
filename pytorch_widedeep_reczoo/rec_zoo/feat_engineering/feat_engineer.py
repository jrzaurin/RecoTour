from typing import List

import pandas as pd
from pytorch_widedeep.utils import LabelEncoder


class FeatureEngineer:
    def __init__(
        self,
        title_col: str,
        genre_col: str,
        cat_cols: List[str],
        genre_separator: str = "|",
    ):
        self.title_col = title_col
        self.clean_title_col = f"{title_col}_clean"
        self.genre_col = genre_col
        self.cat_cols = cat_cols
        self.genre_separator = genre_separator
        self.max_genres = None

        self.label_encoder = LabelEncoder(columns_to_encode=self.cat_cols)

    def fit(self, df: pd.DataFrame) -> "FeatureEngineer":
        df = self._process_dataframe(df)
        self.label_encoder.fit(df)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._process_dataframe(df)
        df = self.label_encoder.transform(df)
        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def _process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = self._extract_year_and_title(df)
        df = self._add_title_word_count(df)
        df = self._format_genres(df)
        return df

    def _extract_year_and_title(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["year"] = df[self.title_col].str.extract(r"\((\d{4})\)")
        df[self.clean_title_col] = df[self.title_col].str.replace(
            r"\s*\(\d{4}\)\s*$", "", regex=True
        )
        df.attrs["year_title_extracted"] = True
        return df

    def _add_title_word_count(self, df: pd.DataFrame) -> pd.DataFrame:
        if not df.attrs.get("year_title_extracted"):
            raise ValueError(
                "Must run _extract_year_and_title before _add_title_word_count"
            )
        df["title_word_count"] = df[self.clean_title_col].apply(
            lambda x: len(str(x).split())
        )
        return df

    def _format_genres(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df[self.genre_col] = (
            df[self.genre_col].str.lower().str.replace(self.genre_separator, "_")
        )
        return df
