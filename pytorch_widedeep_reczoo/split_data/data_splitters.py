from typing import Tuple, Optional

import pandas as pd


def temporal_split(
    data: pd.DataFrame, time_column: str, train_size: float = 0.8, val_size: float = 0.1
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # We can affor to mutate the data
    data = data.sort_values(time_column).reset_index(drop=True)
    n = data.shape[0]
    train_end = int(n * train_size)
    val_end = int(n * (train_size + val_size))
    train = data.iloc[:train_end].reset_index(drop=True)
    val = data.iloc[train_end:val_end].reset_index(drop=True)
    test = data.iloc[val_end:].reset_index(drop=True)
    return train, val, test


def last_interaction_split(
    data: pd.DataFrame, user_column: str, n_interactions: int = 3
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = data.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    users = data[user_column].unique()
    train_idx, val_idx, test_idx = [], [], []
    for u in users:
        u_data = data[data[user_column] == u]
        if u_data.shape[0] >= n_interactions:
            train_idx.extend(u_data.index[:-2])
            val_idx.append(u_data.index[-2])
            test_idx.append(u_data.index[-1])
        else:
            train_idx.extend(u_data.index)
    train = data.iloc[train_idx].reset_index(drop=True)
    val = data.iloc[val_idx].reset_index(drop=True)
    test = data.iloc[test_idx].reset_index(drop=True)
    return train, val, test


def last_interaction_split_with_negatives(
    data: pd.DataFrame,
    user_column: str,
    item_column: str,
    time_column: str,
    n_interactions: int = 3,
    n_negatives: int = 9,
    category_column: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = data.sort_values([user_column, time_column]).reset_index(drop=True)
    users = data[user_column].unique()
    train_data, val_data, test_data = [], [], []
    for user in users:
        user_data = data[data[user_column] == user]
        n_user_interactions = len(user_data)
        if n_user_interactions >= n_interactions:
            train_data.append(user_data.iloc[:-2])
            val_data.append(user_data.iloc[-2:-1])
            test_data.append(user_data.iloc[-1:])
            train_items = set(user_data.iloc[:-2][item_column])
            all_items = set(data[item_column])
            unseen_items = list(all_items - train_items)
            if category_column:
                user_categories = set(user_data[category_column])
                unseen_items = [
                    item
                    for item in unseen_items
                    if data[data[item_column] == item][category_column].iloc[0]
                    in user_categories
                ]
            if len(unseen_items) < 2 * n_negatives:
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
            test_negatives = pd.DataFrame(
                {
                    user_column: [user] * n_negatives,
                    item_column: unseen_items[n_negatives : 2 * n_negatives],
                    "target": [0] * n_negatives,
                }
            )
            val_data.append(val_negatives)
            test_data.append(test_negatives)
        else:
            # If user has less than n_interactions, add all to train
            train_data.append(user_data)
    train = pd.concat(train_data, ignore_index=True)
    val = pd.concat(val_data, ignore_index=True)
    test = pd.concat(test_data, ignore_index=True)
    return train, val, test
