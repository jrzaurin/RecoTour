import shutil
from pathlib import Path

import pandas as pd
import pytest

from rec_tools.split_data.data_splitters import (
    TemporalSplitter,
    LastInteractionSplitter,
    LastInteractionSequenceSplitter,
    LastPositiveInteractionWithNegativeSamplesSplitter,
)


@pytest.fixture(scope="session", autouse=True)
def cleanup_after_all_tests():
    yield
    splits_dir = Path("tests/train_val_test_splits")
    if splits_dir.exists():
        shutil.rmtree(splits_dir)


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

    # Save data to temporary directory structure
    test_dir = tmp_path / "train_val_test_splits" / "movielens_splits"
    test_dir.mkdir(parents=True, exist_ok=True)
    data_path = test_dir / "full_train.csv"
    data.to_csv(data_path, index=False)

    yield str(data_path)

    # No cleanup needed as tmp_path is automatically cleaned up by pytest


@pytest.mark.parametrize("sample_column", ["category", None])
def test_last_positive_interaction_splitter(sample_data, tmp_path, sample_column):
    # Initialize splitter
    splitter = LastPositiveInteractionWithNegativeSamplesSplitter(
        data_path=sample_data,
        user_column="user_id",
        item_column="item_id",
        item_feat_columns=["category"],
        time_column="timestamp",
        dataset="movielens",
        sample_column=sample_column,
        n_negatives=2,
        target_column="rating",
        positive_target=5,
        n_cores=1,
    )

    # Override paths
    splitter.root_dir = tmp_path / "train_val_test_splits"
    splitter.read_path = splitter.root_dir / "movielens_splits"
    splitter.save_path = splitter.root_dir / "lpi_movielens_splits"

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
def test_last_positive_interaction_splitter_val(sample_data, tmp_path):
    # Initialize splitter
    splitter = LastPositiveInteractionWithNegativeSamplesSplitter(
        user_column="user_id",
        item_column="item_id",
        item_feat_columns=["category"],
        time_column="timestamp",
        dataset="movielens",
        n_negatives=2,
        target_column="rating",
        positive_target=5,
        n_cores=1,
    )

    # Override the root_dir to use tmp_path
    splitter.root_dir = tmp_path / "train_val_test_splits"
    splitter.read_path = splitter.root_dir / "movielens_splits"
    splitter.save_path = splitter.root_dir / "lpi_movielens_splits"

    # Split data
    train, val = splitter.split()

    # Assertions
    assert len(val) == 12  # 4 users * 3 interactions (1 positive + 2 negative)

    # Check positive interactions in val set
    positive_vals = val[val["rating"] == 5]
    assert len(positive_vals) == 4

    # Check negative interactions in val set
    negative_vals = val[val["rating"] == 0]
    assert len(negative_vals) == 8  # 2 negative samples per user

    # Verify data integrity
    original_data = pd.read_csv(sample_data)
    assert set(train.columns) == set(original_data.columns)
    assert set(val.columns) == set(original_data.columns)

    # Verify files were saved in the temporary directory
    splits_dir = tmp_path / "train_val_test_splits" / "lpi_movielens_splits"
    assert (splits_dir / "train.csv").exists()
    assert (splits_dir / "val.csv").exists()

    # check that per user id, no negative (0) interaction in the val set is
    # in the train set
    for user_id in val["user_id"].unique():
        user_train = train[train["user_id"] == user_id]
        user_val = val[val["user_id"] == user_id]
        for item_id in user_val["item_id"].values:
            assert item_id not in user_train["item_id"].values


@pytest.mark.parametrize(
    "sample_data",
    ["train_val_test_splits/movielens_splits/full_train.csv"],
    indirect=True,
)
def test_temporal_splitter(sample_data, tmp_path):
    # Initialize splitter
    splitter = TemporalSplitter(
        dataset="movielens",
        user_column="user_id",
        item_column="item_id",
        time_column="timestamp",
        train_size=0.6,
    )

    # Override paths
    splitter.root_dir = tmp_path / "train_val_test_splits"
    splitter.read_path = splitter.root_dir / "movielens_splits"
    splitter.save_path = splitter.root_dir / "ts_movielens_splits"

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

    # Verify files were saved in the temporary directory
    splits_dir = tmp_path / "train_val_test_splits" / "ts_movielens_splits"
    assert (splits_dir / "train.csv").exists()
    assert (splits_dir / "val.csv").exists()


@pytest.mark.parametrize(
    "sample_data",
    ["train_val_test_splits/movielens_splits/full_train.csv"],
    indirect=True,
)
def test_last_interaction_sequence_splitter(sample_data, tmp_path):
    # Initialize splitter
    splitter = LastInteractionSequenceSplitter(
        dataset="movielens",
        user_column="user_id",
        item_column="item_id",
        time_column="timestamp",
        rating_column="rating",
        sequence_length=3,
    )

    # Override paths
    splitter.root_dir = tmp_path / "train_val_test_splits"
    splitter.read_path = splitter.root_dir / "movielens_splits"
    splitter.save_path = splitter.root_dir / "lis_movielens_splits"

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

    # Verify files were saved in the temporary directory
    splits_dir = tmp_path / "train_val_test_splits" / "lis_movielens_splits"
    assert (splits_dir / "train.csv").exists()
    assert (splits_dir / "val.csv").exists()


def test_last_interaction_splitter(sample_data, tmp_path):
    # Initialize splitter with the temporary directory
    splitter = LastInteractionSplitter(
        dataset="movielens",
        user_column="user_id",
        item_column="item_id",
        time_column="timestamp",
    )

    # Temporarily override the root_dir to use tmp_path
    splitter.root_dir = tmp_path / "train_val_test_splits"
    splitter.read_path = splitter.root_dir / "movielens_splits"
    splitter.save_path = splitter.root_dir / "li_movielens_splits"

    # Perform split
    train, val = splitter.split()

    # Verify that each user has exactly one interaction in validation set
    user_counts_val = val["user_id"].value_counts()
    assert all(count == 1 for count in user_counts_val)

    # Verify temporal ordering - last interaction per user is in validation
    original_data = pd.read_csv(sample_data)
    for user_id in val["user_id"].unique():
        user_data = original_data[original_data["user_id"] == user_id]
        user_val = val[val["user_id"] == user_id]
        assert user_val.iloc[0]["timestamp"] == user_data["timestamp"].max()

    # Verify data integrity
    assert set(train.columns) == set(original_data.columns)
    assert set(val.columns) == set(original_data.columns)

    # Verify files were saved in the temporary directory
    splits_dir = tmp_path / "train_val_test_splits" / "li_movielens_splits"
    assert (splits_dir / "train.csv").exists()
    assert (splits_dir / "val.csv").exists()
