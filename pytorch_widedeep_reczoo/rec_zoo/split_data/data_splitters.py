import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Literal, Optional, Tuple

import pandas as pd


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
    """

    def __init__(
        self,
        dataset: Literal["movielens", "amazon"],
        user_column: str,
        item_column: str,
        val_filename: str = "val.csv",
    ):
        self.dataset = dataset
        self.user_column = user_column
        self.item_column = item_column
        self.root_dir = Path("train_val_test_splits")
        self.save_path = self.root_dir / f"{dataset}_splits"
        self.val_filename = val_filename

    def _ensure_save_path(self):
        if not self.save_path.exists():
            self.save_path.mkdir(parents=True, exist_ok=True)

    def _load_data(self, filename: str) -> pd.DataFrame:
        return pd.read_csv(self.save_path / filename)

    def _save_splits(self, train: pd.DataFrame, val: pd.DataFrame):
        self._ensure_save_path()
        train.to_csv(self.save_path / "train.csv", index=False)
        val.to_csv(self.save_path / self.val_filename, index=False)

    @abstractmethod
    def split(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        pass


class LastPositiveInteractionSplitter(BaseDataSplitter):
    """Creates the master train/test split for the recommendation dataset.

    This is typically the first splitter to be used in the data preparation
    pipeline. It creates two datasets:

    1. A test set (saved as 'test.csv'): Contains the last positive
    interaction for each user, along with N negative samples. This set should
    ONLY be used for final model evaluation.
    2. A training set (saved as 'full_train.csv'): Contains all other
    interactions. This will be further split into train/validation sets using
    other splitters.

    The workflow should be:

    1. Use this splitter first to create the master test set
    2. Use other splitters (TemporalSplitter, LastInteractionSequenceSplitter)
    on the 'full_train.csv' to create train/validation splits for model
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
        data_path: str,
        user_column: str,
        item_column: str,
        time_column: str,
        dataset: Literal["movielens", "amazon"],
        n_negatives: int = 9,
        sample_column: Optional[str] = None,
        target_column: str = "rating",
        positive_target: int = 5,
    ):
        super().__init__(dataset, user_column, item_column, val_filename="test.csv")
        self.data_path = data_path
        self.time_column = time_column
        self.n_negatives = n_negatives
        self.sample_column = sample_column
        self.target_column = target_column
        self.positive_target = positive_target

    def split(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        data = pd.read_csv(self.data_path)
        data = data.sort_values([self.user_column, self.time_column]).reset_index(
            drop=True
        )

        users = data[self.user_column].unique()
        test_data: List[pd.Series] = []
        train_data: List[pd.DataFrame] = []

        for user in users:
            user_data = data[data[self.user_column] == user]

            last_interaction = user_data[
                user_data[self.target_column] == self.positive_target
            ].iloc[-1]
            last_interaction_idx = last_interaction.name

            test_data.append(last_interaction)
            user_data_wo_last_interaction = user_data.drop(index=last_interaction_idx)
            train_data.append(user_data_wo_last_interaction)

            user_items = set(user_data[self.item_column])
            all_items = set(data[self.item_column])
            unseen_items = list(all_items - user_items)

            if self.sample_column:
                user_categories = set(user_data[self.sample_column])
                unseen_items = [
                    item
                    for item in unseen_items
                    if data[data[self.item_column] == item][self.sample_column].iloc[0]
                    in user_categories
                ]

            # If not enough unseen items, fallback to all unseen items
            if len(unseen_items) < self.n_negatives:
                unseen_items = list(all_items - user_items)

            negative_items = random.sample(
                unseen_items, min(self.n_negatives, len(unseen_items))
            )

            for neg_item in negative_items:
                negative_interaction = last_interaction.copy()
                negative_interaction[self.item_column] = neg_item
                negative_interaction[self.target_column] = 0
                test_data.append(negative_interaction)

        test = pd.DataFrame(test_data).reset_index(drop=True)
        train = pd.concat(train_data, ignore_index=True)

        self._save_splits(train, test)

        return train, test


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
        super().__init__(dataset, user_column, item_column)
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
        super().__init__(dataset, user_column, item_column)
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

        for user in users:
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
    last_interaction_sequence_splitter = LastInteractionSequenceSplitter(
        dataset="movielens",
        user_column="user_id",
        item_column="item_id",
        time_column="timestamp",
        rating_column="rating",
        sequence_length=5,
    )
    last_interaction_sequence_splitter.split()

    temporal_splitter = TemporalSplitter(
        dataset="movielens",
        user_column="user_id",
        item_column="item_id",
        time_column="timestamp",
        train_size=0.8,
    )
    temporal_splitter.split()

    last_positive_interaction_splitter = LastPositiveInteractionSplitter(
        data_path="movielens",
        user_column="user_id",
        item_column="item_id",
        time_column="timestamp",
        dataset="movielens",
        n_negatives=9,
        sample_column="category",
        target_column="rating",
        positive_target=5,
    )
    last_positive_interaction_splitter.split()
