from datetime import datetime

import pandas as pd
import pytest

from rec_zoo.feat_engineering.item_dynamic_feats import ItemDynamicFeatures


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    data = {
        "user_id": [1, 2, 3, 1, 2, 3],
        "item_id": [101, 101, 101, 102, 102, 103],
        "rating": [4.0, 3.0, 5.0, 2.0, 4.0, 3.0],
        "timestamp": [
            datetime(2023, 1, 1),
            datetime(2023, 1, 15),
            datetime(2023, 1, 30),
            datetime(2023, 1, 1),
            datetime(2023, 1, 10),
            datetime(2023, 1, 5),
        ],
        "gender": ["F", "M", "F", "F", "M", "M"],
        "occupation": [
            "student",
            "engineer",
            "student",
            "teacher",
            "engineer",
            "student",
        ],
        "zipcode": ["12345", "12345", "23456", "12345", "34567", "23456"],
        "age": [25, 30, 25, 35, 30, 28],
    }
    return pd.DataFrame(data)


def test_compute_features(sample_data):
    """Test the main compute_features method."""
    idf = ItemDynamicFeatures()
    features = idf.compute_features(sample_data)

    # Check if all expected columns are present
    expected_columns = [
        "total_ratings",
        "unique_ratings",
        "unique_users",
        "rating_median",
        "rating_mean",
        "rating_std",
        "rating_iqr",
        "occupation_1",
        "occupation_2",
        "occupation_3",
        "age_1",
        "age_2",
        "age_3",
        "female_viewers",
        "male_viewers",
        "mean_days_between_ratings",
        "rating_recency_days",
        "rating_timespan_days",
    ]
    assert all(col in features.columns for col in expected_columns)

    item_id_101_features = features[features.item_id == 101]

    # Check specific values for item 101
    assert item_id_101_features["total_ratings"].values[0] == 3
    assert item_id_101_features["unique_users"].values[0] == 3
    assert item_id_101_features["rating_mean"].values[0] == 4.0
    assert item_id_101_features["female_viewers"].values[0] == 2
    assert item_id_101_features["male_viewers"].values[0] == 1
    assert item_id_101_features["occupation_1"].values[0] == "student"
    assert item_id_101_features["age_1"].values[0] == "25"


def test_compute_time_features(sample_data):
    """Test the time features computation."""
    idf = ItemDynamicFeatures()
    time_features = idf._compute_time_features(sample_data)

    # Check if all time-related columns are present
    expected_columns = [
        "mean_days_between_ratings",
        "rating_recency_days",
        "rating_timespan_days",
    ]
    assert all(col in time_features.columns for col in expected_columns)

    # Check specific values for item 101
    assert time_features.loc[101, "rating_timespan_days"] == 29.0  # Jan 30 - Jan 1
    assert time_features.loc[101, "mean_days_between_ratings"] == pytest.approx(
        14.5, rel=0.1
    )


def test_gender_counts(sample_data):
    """Test gender count calculations."""
    idf = ItemDynamicFeatures()
    features = idf.compute_features(sample_data)

    item_id_101_features = features[features.item_id == 101]
    item_id_102_features = features[features.item_id == 102]

    # Check gender counts for item 101
    assert item_id_101_features["female_viewers"].values[0] == 2
    assert item_id_101_features["male_viewers"].values[0] == 1

    # Check gender counts for item 102
    assert item_id_102_features["female_viewers"].values[0] == 1
    assert item_id_102_features["male_viewers"].values[0] == 1


def test_demographic_modes(sample_data):
    """Test demographic mode calculations."""
    idf = ItemDynamicFeatures()
    features = idf.compute_features(sample_data)

    item_id_101_features = features[features.item_id == 101]

    # Check ranked demographics for item 101
    assert item_id_101_features["occupation_1"].values[0] == "student"
    assert item_id_101_features["occupation_2"].values[0] == "engineer"
    assert item_id_101_features["age_1"].values[0] == "25"
    assert item_id_101_features["age_2"].values[0] == "30"


def test_rating_statistics(sample_data):
    """Test rating statistics calculations."""
    idf = ItemDynamicFeatures()
    features = idf.compute_features(sample_data)

    item_id_101_features = features[features.item_id == 101]

    # Check rating statistics for item 101
    assert item_id_101_features["rating_mean"].values[0] == 4.0
    assert item_id_101_features["rating_median"].values[0] == 4.0
    assert item_id_101_features["unique_ratings"].values[0] == 3
    assert item_id_101_features["rating_iqr"].values[0] == 1.0
