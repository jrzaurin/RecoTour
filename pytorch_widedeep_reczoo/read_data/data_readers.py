from typing import Dict, Optional
from pathlib import Path

import pandas as pd


def read_amazon_reviews(
    path: str,
) -> pd.DataFrame:
    read_fname = "movies_and_tv.json.gz"
    save_fname = "amazon_movies_and_tv_reviews.csv"
    df = pd.read_json(Path(path) / read_fname, lines=True)
    df = df[["reviewerID", "asin", "overall", "reviewText", "unixReviewTime"]]
    df.columns = ["user_id", "item_id", "rating", "review", "timestamp"]
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    df.to_csv(Path(path) / save_fname, index=False)
    print(f"File {save_fname} saved in {path}")
    return df


class MovieLensReader:

    COLUMN_NAMES = {
        "movies": ["movieId", "title", "genres"],
        "ratings": ["userId", "movieId", "rating", "timestamp"],
        "users": ["userId", "gender", "age", "occupation", "zipcode"],
    }
    SAVE_FILENAME = "movielens_ratings_with_info.csv"

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)

    def read_dat_file(
        self, filename: str, columns: Optional[list] = None
    ) -> pd.DataFrame:
        file_path = self.data_dir / f"{filename}.dat"
        columns = self.COLUMN_NAMES[filename]
        df = pd.read_csv(
            file_path, sep="::", engine="python", encoding="latin-1", names=columns
        )
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
        return df

    def read_all(self) -> Dict[str, pd.DataFrame]:
        data: Dict[str, pd.DataFrame] = {}
        for file_type in self.COLUMN_NAMES.keys():
            file_path = self.data_dir / f"{file_type}.dat"
            if file_path.exists():
                data[file_type] = self.read_dat_file(file_type)
        return data

    def read_ratings_with_info(self) -> pd.DataFrame:
        ratings = self.read_dat_file("ratings")
        movies = self.read_dat_file("movies")
        users = self.read_dat_file("users")
        full_data = ratings.merge(movies, on="movieId", how="left").merge(
            users, on="userId", how="left"
        )
        full_data.to_csv(self.data_dir / self.SAVE_FILENAME, index=False)
        print(f"File {self.SAVE_FILENAME} saved in {self.data_dir}")
        return full_data


if __name__ == "__main__":
    amazon_movies_and_tv_reviews = read_amazon_reviews("data/amtv-5")
    movielens_ratings_with_info = MovieLensReader("data/ml-1m").read_ratings_with_info()
