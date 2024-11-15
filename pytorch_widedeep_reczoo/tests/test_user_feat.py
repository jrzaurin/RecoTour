from datetime import datetime

import pandas as pd
import pytest

from rec_zoo.feat_engineering.user_dynamic_feats import UserDynamicFeatures


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    data = {
        "user_id": [1, 1, 1, 2, 2, 3],
        "item_id": [101, 102, 103, 201, 202, 301],
        "rating": [4.0, 3.0, 5.0, 2.0, 4.0, 3.0],
        "timestamp": [
            datetime(2023, 1, 1),
            datetime(2023, 1, 15),
            datetime(2023, 1, 30),
            datetime(2023, 1, 1),
            datetime(2023, 1, 10),
            datetime(2023, 1, 5),
        ],
        "genres": ["action", "comedy", "action", "drama", "comedy", "action"],
    }
    return pd.DataFrame(data)


def test_compute_features(sample_data):
    """Test the main compute_features method."""
    udf = UserDynamicFeatures()
    features = udf.compute_features(sample_data)

    # Check if all expected columns are present
    expected_columns = [
        "total_movies",
        "unique_movies",
        "total_ratings",
        "rating_mean",
        "rating_std",
        "rating_median",
        "rating_iqr",
        "mean_days_between_views",
        "viewing_recency_days",
        "viewing_timespan_days",
        "viewing_frequency_weekly",
        "favorite_genre_1",
        "favorite_genre_2",
        "favorite_genre_3",
        "unique_genres_count",
    ]
    assert all(col in features.columns for col in expected_columns)

    features_user_1 = features[features.user_id == 1]

    # Check specific values for user 1
    assert features_user_1["total_movies"].values[0] == 3
    assert features_user_1["unique_movies"].values[0] == 3
    assert features_user_1["rating_mean"].values[0] == 4.0
    assert features_user_1["unique_genres_count"].values[0] == 2
    assert features_user_1["favorite_genre_1"].values[0] == "action"


def test_compute_time_features(sample_data):
    """Test the time features computation."""
    udf = UserDynamicFeatures()
    time_features = udf._compute_time_features(sample_data)

    # Check if all time-related columns are present
    expected_columns = [
        "mean_days_between_views",
        "viewing_recency_days",
        "viewing_timespan_days",
        "viewing_frequency_weekly",
    ]
    assert all(col in time_features.columns for col in expected_columns)

    # Check specific values for user 1
    assert time_features.loc[1, "viewing_timespan_days"] == 29.0  # Jan 30 - Jan 1
    assert time_features.loc[1, "viewing_frequency_weekly"] == pytest.approx(
        0.75, rel=0.1
    )  # 3 views over ~4 weeks


def test_compute_genre_preferences(sample_data):
    """Test the genre preferences computation."""
    udf = UserDynamicFeatures()
    genre_features = udf._compute_genre_preferences(sample_data)

    # Check if all genre-related columns are present
    expected_columns = [
        "favorite_genre_1",
        "favorite_genre_2",
        "favorite_genre_3",
        "unique_genres_count",
    ]
    assert all(col in genre_features.columns for col in expected_columns)

    # Check specific values
    assert genre_features.loc[1, "favorite_genre_1"] == "action"
    assert genre_features.loc[1, "unique_genres_count"] == 2
    assert genre_features.loc[2, "unique_genres_count"] == 2
