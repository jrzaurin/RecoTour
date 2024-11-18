import random
import multiprocessing as mp
from abc import ABC, abstractmethod
from typing import List, Tuple, Literal, Optional
from pathlib import Path
from functools import partial

import pandas as pd
from tqdm import tqdm


class BaseDataSplitter(ABC):
    """Abstract base class for data splitting strategies.

    This class provides the basic functionality for loading, saving, and
    splitting datasets for recommendation systems.

    Args:
        dataset (Literal["movielens", "amazon"]):
            Name of the dataset to split
        user_column (str):
            Name of the column containing user IDs
        item_column (str):
            Name of the column containing item IDs
        val_filename (str, optional):
            Name of the validation file. Defaults to "val.csv"
        split_prefix (str, optional):
            Prefix to add to the save path. Defaults to ""
    """

    def __init__(
        self,
        dataset: Literal["movielens", "amazon"],
        user_column: str,
        item_column: str,
        train_filename: str = "train.csv",
        val_filename: str = "val.csv",
        split_prefix: str = "",
    ):
        self.dataset = dataset
        self.user_column = user_column
        self.item_column = item_column
        self.root_dir = Path("train_val_test_splits")
        self.read_path = self.root_dir / f"{dataset}_splits"
        prefix = f"{split_prefix}_" if split_prefix else ""
        self.save_path = self.root_dir / f"{prefix}{dataset}_splits"
        self.train_filename = train_filename
        self.val_filename = val_filename

    def _ensure_save_path(self):
        if not self.save_path.exists():
            self.save_path.mkdir(parents=True, exist_ok=True)

    def _load_data(self, filename: str) -> pd.DataFrame:
        return pd.read_csv(self.read_path / filename)

    def _save_splits(self, train: pd.DataFrame, val: pd.DataFrame):
        self._ensure_save_path()
        train.to_csv(self.save_path / self.train_filename, index=False)
        val.to_csv(self.save_path / self.val_filename, index=False)

    @abstractmethod
    def split(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        pass


class LastPositiveInteractionWithNegativeSamplesSplitter(BaseDataSplitter):
    """Creates the master train/test split for the recommendation dataset.

    This is the first splitter to be used in the data preparation pipeline. It
    creates two datasets:

    1. A test set (saved as 'test.csv'): Contains the last positive
    interaction for each user, along with N negative samples. This set should
    ONLY be used for final model evaluation.
    2. A training set (saved as 'full_train.csv'): Contains all other
    interactions. This will be further split into train/validation sets using
    other splitters.

    The workflow should be:

    1. Use this splitter first to create the master test set
    2. Use it again (with the appropiate params), or the other splitters
    (TemporalSplitter, LastInteractionSequenceSplitter) on
    the 'full_train.csv' to create train/validation splits for model
    development.

    Args:
        data_path (str):
            Path to the input data file
        user_column (str):
            Name of the column containing user IDs
        item_column (str):
            Name of the column containing item IDs
        time_column (str):
            Name of the column containing timestamps
        dataset (Literal["movielens", "amazon"]):
            Name of the dataset
        n_negatives (int, optional):
            Number of negative samples per positive test case. Defaults to 9
        sample_column (str, optional):
            Column to use for stratified sampling. Defaults to None
        target_column (str, optional):
            Name of the target column. Defaults to "rating"
        positive_target (int, optional):
            Value that indicates a positive interaction. Defaults to 5
    """

    def __init__(
        self,
        user_column: str,
        item_column: str,
        time_column: str,
        item_feat_columns: List[str],
        dataset: Literal["movielens", "amazon"],
        data_path: Optional[str] = None,
        n_negatives: int = 9,
        sample_column: Optional[str] = None,
        target_column: str = "rating",
        positive_target: int = 5,
        n_cores: int | None = None,
    ):
        train_filename = "full_train.csv" if data_path else "train.csv"
        val_filename = "test.csv" if data_path else "val.csv"
        split_prefix = "" if data_path else "lpi"
        super().__init__(
            dataset,
            user_column,
            item_column,
            train_filename=train_filename,
            val_filename=val_filename,
            split_prefix=split_prefix,
        )
        self.item_feat_columns = item_feat_columns
        self.data_path = data_path
        self.time_column = time_column
        self.n_negatives = n_negatives
        self.sample_column = sample_column
        self.target_column = target_column
        self.positive_target = positive_target
        self.n_cores = n_cores

    def split(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if self.data_path:
            data = pd.read_csv(self.data_path)
        else:
            data = self._load_data("full_train.csv")

        data = data.sort_values([self.user_column, self.time_column]).reset_index(
            drop=True
        )
        users = data[self.user_column].unique()

        partial_process_user = partial(self._process_user, data=data)

        n_jobs = self.n_cores if self.n_cores else mp.cpu_count()
        with mp.Pool(n_jobs) as pool:
            results = list(
                tqdm(
                    pool.imap(partial_process_user, users),
                    total=len(users),
                    desc=f"Processing users for LPI ({len(users)} users)",
                )
            )

        train_data = [result[0] for result in results if result[0] is not None]
        test_data = [result[1] for result in results if result[1] is not None]

        train = pd.concat(train_data, ignore_index=True)
        test = pd.concat(test_data, ignore_index=True)

        self._save_splits(train, test)

        return train, test

    def _process_user(
        self, user: str, data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        user_data = data[data[self.user_column] == user]
        if user_data.empty:
            return None, None

        positive_interactions = user_data[
            user_data[self.target_column] == self.positive_target
        ]
        if len(positive_interactions) == 0:
            return user_data, None

        last_positive_interaction, user_data_wo_last_interaction = (
            self._split_user_data(user_data, positive_interactions)
        )
        negative_samples = self._generate_negative_samples(
            data, user_data, last_positive_interaction
        )
        positive_sample = pd.DataFrame([last_positive_interaction.to_dict()])
        user_id_test_data = pd.concat(
            [positive_sample, negative_samples], ignore_index=True
        )
        return user_data_wo_last_interaction, user_id_test_data

    def _split_user_data(
        self, user_data: pd.DataFrame, positive_interactions: pd.DataFrame
    ) -> Tuple[pd.Series, pd.DataFrame]:
        last_positive_interaction = positive_interactions.iloc[-1]
        user_data_wo_last_interaction = user_data.drop(last_positive_interaction.name)
        return last_positive_interaction, user_data_wo_last_interaction

    def _generate_negative_samples(
        self,
        data: pd.DataFrame,
        user_data: pd.DataFrame,
        last_positive_interaction: pd.Series,
    ) -> pd.DataFrame:
        user_items = set(user_data[self.item_column])

        item_features_lookup = (
            data[[self.item_column] + self.item_feat_columns]
            .drop_duplicates()
            .set_index(self.item_column)
            .to_dict("index")
        )

        all_negative_items = list(set(item_features_lookup.keys()) - user_items)

        if self.sample_column:
            user_categories = set(user_data[self.sample_column])
            filtered_items = [
                item
                for item in all_negative_items
                if item_features_lookup[item][self.sample_column] in user_categories
            ]

            negative_pool = (
                filtered_items
                if len(filtered_items) >= self.n_negatives
                else all_negative_items
            )
        else:
            negative_pool = all_negative_items

        negative_items = random.sample(
            negative_pool, min(self.n_negatives, len(negative_pool))
        )

        negative_samples = pd.DataFrame(
            [
                {**item_features_lookup[item], self.item_column: item}
                for item in negative_items
            ]
        )

        negative_samples[self.target_column] = 0

        cols_to_copy = [
            col
            for col in data.columns
            if col
            not in {self.item_column, self.target_column, *self.item_feat_columns}
        ]
        for col in cols_to_copy:
            negative_samples[col] = last_positive_interaction[col]

        return negative_samples


class LastInteractionSplitter(BaseDataSplitter):
    """Splits data into train and validation sets using the last interaction for validation.

    This splitter uses all but the last interaction per user for training and
    the last interaction for validation.

    Args:
        dataset (Literal["movielens", "amazon"]):
            Name of the dataset
        user_column (str):
            Name of the column containing user IDs
        item_column (str):
            Name of the column containing item IDs
        time_column (str):
            Name of the column containing timestamps
    """

    def __init__(
        self,
        dataset: Literal["movielens", "amazon"],
        user_column: str,
        item_column: str,
        time_column: str,
    ):
        super().__init__(
            dataset,
            user_column,
            item_column,
            split_prefix="li",
        )
        self.time_column = time_column

    def split(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        full_train = self._load_data("full_train.csv")
        full_train = full_train.sort_values(
            [self.user_column, self.time_column]
        ).reset_index(drop=True)
        users = full_train[self.user_column].unique()

        train_data, val_data = [], []

        for user in tqdm(users, desc=f"Processing users for LI ({len(users)} users)"):
            user_data = full_train[full_train[self.user_column] == user]

            if user_data.empty:
                continue

            last_interaction = user_data.iloc[-1]
            user_data_wo_last_interaction = user_data.iloc[:-1]

            train_data.append(user_data_wo_last_interaction)
            val_data.append(last_interaction)

        train = pd.concat(train_data, ignore_index=True)
        val = pd.DataFrame(val_data)

        self._save_splits(train, val)

        return train, val


class TemporalSplitter(BaseDataSplitter):
    """Splits data temporally into train and validation sets.

    Sorts data by timestamp and splits it into training and validation sets
    based on the specified train size ratio.

    Args:
        dataset (Literal["movielens", "amazon"]):
            Name of the dataset
        user_column (str):
            Name of the column containing user IDs
        item_column (str):
            Name of the column containing item IDs
        time_column (str):
            Name of the column containing timestamps
        train_size (float, optional):
            Proportion of data to use for training. Defaults to 0.8
    """

    def __init__(
        self,
        dataset: Literal["movielens", "amazon"],
        user_column: str,
        item_column: str,
        time_column: str,
        train_size: float = 0.8,
    ):
        super().__init__(
            dataset,
            user_column,
            item_column,
            split_prefix="ts",
        )
        self.time_column = time_column
        self.train_size = train_size

    def split(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        full_train = self._load_data("full_train.csv")
        full_train = full_train.sort_values(self.time_column).reset_index(drop=True)

        n = full_train.shape[0]
        train_end = int(n * self.train_size)

        train = full_train.iloc[:train_end].reset_index(drop=True)
        val = full_train.iloc[train_end:].reset_index(drop=True)

        self._save_splits(train, val)

        return train, val


class LastInteractionSequenceSplitter(BaseDataSplitter):
    """Splits data into sequences for sequential recommendation.

    Creates sequences of interactions for each user, using the last interaction as the
    target for validation. Handles cases where users have fewer interactions than the
    sequence length by adding padding.

    Args:
        dataset (Literal["movielens", "amazon"]):
            Name of the dataset
        user_column (str):
            Name of the column containing user IDs
        item_column (str):
            Name of the column containing item IDs
        time_column (str):
            Name of the column containing timestamps
        rating_column (str):
            Name of the column containing ratings
        sequence_length (int, optional):
            Length of the interaction sequences. Defaults to 5
    """

    def __init__(
        self,
        dataset: Literal["movielens", "amazon"],
        user_column: str,
        item_column: str,
        time_column: str,
        rating_column: str,
        sequence_length: int = 5,
    ):
        super().__init__(
            dataset,
            user_column,
            item_column,
            split_prefix="lis",
        )
        self.time_column = time_column
        self.rating_column = rating_column
        self.sequence_length = sequence_length

    def split(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        full_train = self._load_data("full_train.csv")
        full_train = full_train.sort_values(
            [self.user_column, self.time_column]
        ).reset_index(drop=True)
        users = full_train[self.user_column].unique()

        train_data, val_data = [], []

        for user in tqdm(users, desc=f"Processing users for LIS ({len(users)} users)"):
            user_data = full_train[full_train[self.user_column] == user]
            n_interactions = len(user_data)

            if n_interactions >= self.sequence_length + 1:
                train_interactions = user_data.iloc[:-1]
                val_interaction = user_data.iloc[-1]

                for i in range(len(train_interactions) - self.sequence_length + 1):
                    item_seq = (
                        train_interactions[self.item_column]
                        .iloc[i : i + self.sequence_length]
                        .tolist()
                    )
                    rating_seq = (
                        train_interactions[self.rating_column]
                        .iloc[i : i + self.sequence_length]
                        .tolist()
                    )
                    train_data.append(
                        {
                            self.user_column: user,
                            "item_sequence": item_seq,
                            "rating_sequence": rating_seq,
                        }
                    )

                val_data.append(
                    {
                        self.user_column: user,
                        "target": val_interaction[self.item_column],
                        "rating": val_interaction[self.rating_column],
                    }
                )
            elif n_interactions >= 1:
                for i in range(n_interactions):
                    item_seq = (
                        user_data[self.item_column]
                        .iloc[i : i + self.sequence_length]
                        .tolist()
                    )
                    rating_seq = (
                        user_data[self.rating_column]
                        .iloc[i : i + self.sequence_length]
                        .tolist()
                    )
                    padding = [0] * (self.sequence_length - len(item_seq))
                    item_seq = padding + item_seq
                    rating_seq = padding + rating_seq
                    train_data.append(
                        {
                            self.user_column: user,
                            "item_sequence": item_seq,
                            "rating_sequence": rating_seq,
                        }
                    )

        train = pd.DataFrame(train_data)
        val = pd.DataFrame(val_data)

        self._save_splits(train, val)

        return train, val


if __name__ == "__main__":

    last_positive_interaction_splitter = (
        LastPositiveInteractionWithNegativeSamplesSplitter(
            user_column="user_id",
            item_column="item_id",
            time_column="timestamp",
            item_feat_columns=["title", "genres"],
            dataset="movielens",
            data_path="data/ml-1m/movielens_ratings_with_info.csv",
            n_negatives=99,
            sample_column="genres",
        )
    )
    full_train, test = last_positive_interaction_splitter.split()

    last_positive_interaction_splitter_val = (
        LastPositiveInteractionWithNegativeSamplesSplitter(
            user_column="user_id",
            item_column="item_id",
            time_column="timestamp",
            item_feat_columns=["title", "genres"],
            dataset="movielens",
            n_negatives=99,
            sample_column="genres",
        )
    )
    lpi_train, lpi_val = last_positive_interaction_splitter_val.split()

    last_interaction_splitter = LastInteractionSplitter(
        dataset="movielens",
        user_column="user_id",
        item_column="item_id",
        time_column="timestamp",
    )
    li_train, li_val = last_interaction_splitter.split()

    temporal_splitter = TemporalSplitter(
        dataset="movielens",
        user_column="user_id",
        item_column="item_id",
        time_column="timestamp",
    )
    t_train, t_val = temporal_splitter.split()

    last_interaction_sequence_splitter = LastInteractionSequenceSplitter(
        dataset="movielens",
        user_column="user_id",
        item_column="item_id",
        time_column="timestamp",
        rating_column="rating",
        sequence_length=10,
    )
    lis_train, lis_val = last_interaction_sequence_splitter.split()
