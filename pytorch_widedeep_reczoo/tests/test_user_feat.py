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
