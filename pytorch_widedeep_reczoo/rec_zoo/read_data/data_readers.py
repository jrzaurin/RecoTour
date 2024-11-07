# in pple this file could be run on the root of the project. Also could be
# run only once, although if run again the classes will simply print that the
# files already exist and return the dataframes. to run, make sure the data
# directory exists and contains the raw data. Also make sure you properly
# -----------------------------------------------------------------------------
# define the PATH variable for example:
# export PYTHONPATH=/Users/youruser/ml_projects/RecoTour/pytorch_widedeep_reczoo
# then to run the script:
# python rec_zoo/read_data/data_readers.py
# -----------------------------------------------------------------------------

from pathlib import Path
from typing import Optional

import pandas as pd


class AmazonReviewsReader:
    """Reader class for Amazon Movies and TV Reviews dataset.

    This class handles reading and processing the Amazon Movies and TV Reviews dataset,
    which contains user reviews and ratings for movies and TV shows from Amazon.

    Args:
        data_dir (str):
            Directory path where the dataset is located
        replace (bool, optional):
            If True, overwrites existing processed file. Defaults to False.
    """

    def __init__(self, data_dir: str, replace: bool = False):
        self.data_dir = Path(data_dir)
        self.read_fname = "movies_and_tv.json.gz"
        self.save_fname = "amazon_movies_and_tv_reviews.csv"
        self.replace = replace

    def read_reviews(self) -> pd.DataFrame:
        """Read and process the Amazon Movies and TV Reviews dataset.

        Returns:
            pd.DataFrame: Processed DataFrame with columns:
                - user_id: Unique identifier for each reviewer
                - item_id: Product ASIN (Amazon Standard Identification Number)
                - rating: Rating score (1-5)
                - review: Text content of the review
                - timestamp: DateTime of when the review was posted
        """
        save_path = self.data_dir / self.save_fname
        if save_path.exists() and not self.replace:
            print(f"File {self.save_fname} already exists in {self.data_dir}")
            return pd.read_csv(save_path)

        df = pd.read_json(self.data_dir / self.read_fname, lines=True)
        df = df[["reviewerID", "asin", "overall", "reviewText", "unixReviewTime"]]
        df.columns = ["user_id", "item_id", "rating", "review", "timestamp"]
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
        df.to_csv(save_path, index=False)
        print(f"File {self.save_fname} saved in {self.data_dir}")
        return df


class MovieLensReader:
    """Reader class for MovieLens dataset.

    This class handles reading and processing the MovieLens dataset, which includes
    movie ratings along with user and movie metadata.

    Args:
        data_dir (str):
            Directory path where the dataset is located
        replace (bool, optional):
            If True, overwrites existing processed file. Defaults to False.
    """

    COLUMN_NAMES = {
        "movies": ["item_id", "title", "genres"],
        "ratings": ["user_id", "item_id", "rating", "timestamp"],
        "users": ["user_id", "gender", "age", "occupation", "zipcode"],
    }
    SAVE_FILENAME = "movielens_ratings_with_info.csv"

    def __init__(self, data_dir: str, replace: bool = False):
        self.data_dir = Path(data_dir)
        self.replace = replace

    def read_dat_file(
        self, filename: str, columns: Optional[list] = None
    ) -> pd.DataFrame:
        """Read a .dat file from the MovieLens dataset.

        Args:
            filename (str): Name of the file to read (without extension)
            columns (Optional[list], optional): Column names. Defaults to None.

        Returns:
            pd.DataFrame: DataFrame containing the file data with appropriate columns
        """
        file_path = self.data_dir / f"{filename}.dat"
        columns = self.COLUMN_NAMES[filename]
        df = pd.read_csv(
            file_path, sep="::", engine="python", encoding="latin-1", names=columns
        )
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
        return df

    def read_ratings_with_info(self) -> pd.DataFrame:
        """Read and merge all MovieLens data files into a single DataFrame.

        Combines ratings data with movie and user information into a single DataFrame.

        Returns:
            pd.DataFrame: Combined DataFrame with columns from ratings, movies, and users files:
                - user_id: Unique identifier for each user
                - item_id: Unique identifier for each movie
                - rating: Rating score (1-5)
                - timestamp: DateTime of when the rating was made
                - title: Movie title
                - genres: Movie genres
                - gender: User gender
                - age: User age
                - occupation: User occupation
                - zipcode: User zipcode
        """
        save_path = self.data_dir / self.SAVE_FILENAME
        if save_path.exists() and not self.replace:
            print(f"File {self.SAVE_FILENAME} already exists in {self.data_dir}")
            return pd.read_csv(save_path)

        ratings = self.read_dat_file("ratings")
        movies = self.read_dat_file("movies")
        users = self.read_dat_file("users")
        full_data = ratings.merge(movies, on="item_id", how="left").merge(
            users, on="user_id", how="left"
        )
        full_data.to_csv(save_path, index=False)
        print(f"File {self.SAVE_FILENAME} saved in {self.data_dir}")
        return full_data


if __name__ == "__main__":
    amazon_movies_and_tv_reviews = AmazonReviewsReader("data/amtv-5").read_reviews()
    movielens_ratings_with_info = MovieLensReader("data/ml-1m").read_ratings_with_info()
