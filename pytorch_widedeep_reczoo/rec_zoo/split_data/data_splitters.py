import random
from pathlib import Path
from typing import List, Literal, Optional, Tuple

import pandas as pd


def last_positive_interaction_with_negatives(
    data_path: str,
    user_column: str,
    item_column: str,
    time_column: str,
    dataset: Literal["movielens", "amazon"],
    n_negatives: int = 9,
    sample_column: Optional[str] = None,
    target_column: str = "rating",
    positive_target: int = 5,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    data = pd.read_csv(data_path)
    data = data.sort_values([user_column, time_column]).reset_index(drop=True)

    users = data[user_column].unique()
    test_data: List[pd.Series] = []
    train_data: List[pd.DataFrame] = []

    for user in users:
        user_data = data[data[user_column] == user]

        last_interaction = user_data[user_data[target_column] == positive_target].iloc[
            -1
        ]
        last_interaction_idx = last_interaction.name

        test_data.append(last_interaction)
        user_data = user_data.drop(index=last_interaction_idx)

        user_items = set(user_data[item_column])
        all_items = set(data[item_column])
        unseen_items = list(all_items - user_items)

        if sample_column:
            user_categories = set(user_data[sample_column])
            unseen_items = [
                item
                for item in unseen_items
                if data[data[item_column] == item][sample_column].iloc[0]
                in user_categories
            ]

        # If not enough unseen items, fallback to all unseen items
        if len(unseen_items) < n_negatives:
            unseen_items = list(all_items - user_items)

        negative_items = random.sample(
            unseen_items, min(n_negatives, len(unseen_items))
        )

        for neg_item in negative_items:
            negative_interaction = last_interaction.copy()
            negative_interaction[item_column] = neg_item
            negative_interaction[target_column] = 0
            test_data.append(negative_interaction)

    test = pd.DataFrame(test_data).reset_index(drop=True)
    train = pd.concat(train_data, ignore_index=True)

    root_dir = Path("train_val_test_splits")
    save_path = Path(root_dir) / f"{dataset}_splits"
    if not save_path.exists():
        save_path.mkdir(parents=True, exist_ok=True)

    train.to_csv(save_path / "full_train.csv", index=False)
    test.to_csv(save_path / "test.csv", index=False)

    return train, test


def temporal_split(
    dataset: Literal["movielens", "amazon"],
    time_column: str,
    train_size: float = 0.8,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    root_dir = Path("train_val_test_splits")
    save_path = Path(root_dir) / f"{dataset}_splits"

    full_train = pd.read_csv(save_path / "full_train.csv")
    full_train = full_train.sort_values(time_column).reset_index(drop=True)

    n = full_train.shape[0]
    train_end = int(n * train_size)

    train = full_train.iloc[:train_end].reset_index(drop=True)
    val = full_train.iloc[train_end:].reset_index(drop=True)

    train.to_csv(save_path / "train.csv", index=False)
    val.to_csv(save_path / "val.csv", index=False)

    return train, val


def last_interaction_split_with_negatives(
    dataset: Literal["movielens", "amazon"],
    user_column: str,
    item_column: str,
    time_column: str,
    n_interactions: int = 3,
    n_negatives: int = 9,
    category_column: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    root_dir = Path("train_val_test_splits")
    save_path = Path(root_dir) / f"{dataset}_splits"

    full_train = pd.read_csv(save_path / "full_train.csv")
    full_train = full_train.sort_values([user_column, time_column]).reset_index(
        drop=True
    )

    users = full_train[user_column].unique()
    train_data, val_data = [], []
    for user in users:
        user_data = full_train[full_train[user_column] == user]

        n_user_interactions = len(user_data)

        if n_user_interactions >= n_interactions:
            train_data.append(user_data.iloc[:-1])
            val_data.append(user_data.iloc[-1:])

            train_items = set(user_data.iloc[:-1][item_column])
            all_items = set(full_train[item_column])
            unseen_items = list(all_items - train_items)

            if category_column:
                user_categories = set(user_data[category_column])
                unseen_items = [
                    item
                    for item in unseen_items
                    if full_train[full_train[item_column] == item][
                        category_column
                    ].iloc[0]
                    in user_categories
                ]

            if len(unseen_items) < n_negatives:
                unseen_items = list(
                    all_items - train_items
                )  # Fallback to all unseen items

            val_negatives = pd.DataFrame(
                {
                    user_column: [user] * n_negatives,
                    item_column: unseen_items[:n_negatives],
                    "target": [0] * n_negatives,
                }
            )
            val_data.append(val_negatives)
        else:
            # If user has less than n_interactions, add all to train
            train_data.append(user_data)

    train = pd.concat(train_data, ignore_index=True)
    val = pd.concat(val_data, ignore_index=True)

    train.to_csv(save_path / "train.csv", index=False)
    val.to_csv(save_path / "val.csv", index=False)

    return train, val


def last_interaction_sequence_split(
    dataset: Literal["movielens", "amazon"],
    user_column: str,
    item_column: str,
    rating_column: str,
    sequence_length: int = 5,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    root_dir = Path("train_val_test_splits")
    save_path = Path(root_dir) / f"{dataset}_splits"

    full_train = pd.read_csv(save_path / "full_train.csv")
    full_train = full_train.sort_values([user_column, rating_column]).reset_index(
        drop=True
    )
    users = full_train[user_column].unique()

    train_data, val_data = [], []

    for user in users:
        user_data = full_train[full_train[user_column] == user]
        n_interactions = len(user_data)

        if n_interactions >= sequence_length + 1:
            train_interactions = user_data.iloc[:-1]
            val_interaction = user_data.iloc[-1]

            for i in range(len(train_interactions) - sequence_length + 1):
                item_seq = (
                    train_interactions[item_column]
                    .iloc[i : i + sequence_length]
                    .tolist()
                )
                rating_seq = (
                    train_interactions[rating_column]
                    .iloc[i : i + sequence_length]
                    .tolist()
                )
                train_data.append(
                    {
                        user_column: user,
                        "item_sequence": item_seq,
                        "rating_sequence": rating_seq,
                    }
                )

            val_data.append(
                {
                    user_column: user,
                    "target": val_interaction[item_column],
                    "rating": val_interaction[rating_column],
                }
            )
        elif n_interactions >= 1:
            for i in range(n_interactions):
                item_seq = user_data[item_column].iloc[i : i + sequence_length].tolist()
                rating_seq = (
                    user_data[rating_column].iloc[i : i + sequence_length].tolist()
                )
                padding = [0] * (sequence_length - len(item_seq))
                item_seq = padding + item_seq
                rating_seq = padding + rating_seq
                train_data.append(
                    {
                        user_column: user,
                        "item_sequence": item_seq,
                        "rating_sequence": rating_seq,
                    }
                )

    train = pd.DataFrame(train_data)
    val = pd.DataFrame(val_data)

    train.to_csv(save_path / "train.csv", index=False)
    val.to_csv(save_path / "val.csv", index=False)

    return train, val
