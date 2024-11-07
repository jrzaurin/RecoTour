import shutil
from pathlib import Path

import pandas as pd
import pytest

from rec_zoo.split_data.data_splitters import (
    LastInteractionSequenceSplitter,
    LastPositiveInteractionSplitter,
    TemporalSplitter,
)


@pytest.fixture
def sample_data(tmp_path, request):
    # create a dataset with 20 interactions
    data = pd.DataFrame(
        {
            "user_id": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4],
            "item_id": [1, 2, 3, 4, 5, 2, 3, 4, 5, 6, 1, 3, 5, 7, 8, 2, 4, 6, 8, 9],
            "timestamp": pd.date_range(start="2023-01-01", periods=20, freq="D"),
            "rating": [3, 4, 5, 3, 4, 2, 5, 1, 4, 5, 3, 5, 4, 5, 2, 4, 5, 3, 4, 5],
            "category": [
                "A",
                "B",
                "C",
                "A",
                "B",
                "B",
                "C",
                "A",
                "B",
                "C",
                "A",
                "C",
                "B",
                "A",
                "B",
                "B",
                "A",
                "C",
                "B",
                "C",
            ],
        }
    )

    # Get the save_path parameter, default to tmp_path
    save_path = getattr(request, "param", None)

    if save_path:
        # If save_path is specified, save to train_val_test_splits/{dataset}_splits/full_train.csv
        save_dir = Path(save_path).parent
        save_dir.mkdir(parents=True, exist_ok=True)
        data.to_csv(save_path, index=False)
        data_path = save_path
    else:
        # Original behavior: save to tmp_path
        data_path = tmp_path / "test_data.csv"
        data.to_csv(data_path, index=False)

    yield str(data_path)

    # cleanup
    splits_dir = Path("train_val_test_splits")
    if splits_dir.exists():
        shutil.rmtree(splits_dir)


@pytest.mark.parametrize("sample_column", ["category", None])
def test_last_positive_interaction_splitter(sample_data, sample_column):
    # Initialize splitter
    splitter = LastPositiveInteractionSplitter(
        data_path=sample_data,
        user_column="user_id",
        item_column="item_id",
        time_column="timestamp",
        dataset="movielens",
        sample_column=sample_column,
        n_negatives=2,
        target_column="rating",
        positive_target=5,
    )

    # Split data
    train, test = splitter.split()

    # Assertions
    assert len(test) == 12  # 4 users * (1 positive + 2 negative) interactions

    # Check positive interactions in test set
    positive_tests = test[test["rating"] == 5]
    assert len(positive_tests) == 4

    # Check negative interactions in test set
    negative_tests = test[test["rating"] == 0]
    assert len(negative_tests) == 8  # 2 negative samples per user

    # check that per user id, no negative (0) interaction in the test set is
    # in the train set
    for user_id in test["user_id"].unique():
        user_train = train[train["user_id"] == user_id]
        user_test = test[test["user_id"] == user_id]
        for item_id in user_test["item_id"].values:
            assert item_id not in user_train["item_id"].values


@pytest.mark.parametrize(
    "sample_data",
    ["train_val_test_splits/movielens_splits/full_train.csv"],
    indirect=True,
)
def test_temporal_splitter(sample_data):
    # Initialize splitter
    splitter = TemporalSplitter(
        dataset="movielens",
        user_column="user_id",
        item_column="item_id",
        time_column="timestamp",
        train_size=0.6,
    )

    # Perform split
    train, val = splitter.split()

    # Verify split sizes (60% of 20 records)
    assert len(train) == 12
    assert len(val) == 8

    # Verify temporal ordering
    assert train["timestamp"].max() < val["timestamp"].min()

    # Verify data integrity
    original_data = pd.read_csv(sample_data)
    assert set(train.columns) == set(original_data.columns)
    assert set(val.columns) == set(original_data.columns)

    # Verify files were saved
    splits_dir = Path("train_val_test_splits") / "movielens_splits"
    assert (splits_dir / "train.csv").exists()
    assert (splits_dir / "val.csv").exists()


@pytest.mark.parametrize(
    "sample_data",
    ["train_val_test_splits/movielens_splits/full_train.csv"],
    indirect=True,
)
def test_last_interaction_sequence_splitter(sample_data):
    # Initialize splitter
    splitter = LastInteractionSequenceSplitter(
        dataset="movielens",
        user_column="user_id",
        item_column="item_id",
        time_column="timestamp",
        rating_column="rating",
        sequence_length=3,
    )

    # Perform split
    train, val = splitter.split()

    # Basic assertions
    assert "item_sequence" in train.columns
    assert "rating_sequence" in train.columns
    assert "target" in val.columns
    assert "rating" in val.columns

    # Check sequence length
    assert all(len(seq) == 3 for seq in train["item_sequence"])
    assert all(len(seq) == 3 for seq in train["rating_sequence"])

    # Check that validation targets are the last items for users with enough interactions
    original_data = pd.read_csv(sample_data)
    for user_id in val["user_id"].unique():
        user_data = original_data[original_data["user_id"] == user_id]
        assert (
            val[val["user_id"] == user_id]["target"].iloc[0]
            == user_data["item_id"].iloc[-1]
        )
        assert (
            val[val["user_id"] == user_id]["rating"].iloc[0]
            == user_data["rating"].iloc[-1]
        )

    # Verify files were saved
    splits_dir = Path("train_val_test_splits") / "movielens_splits"
    assert (splits_dir / "train.csv").exists()
    assert (splits_dir / "val.csv").exists()
