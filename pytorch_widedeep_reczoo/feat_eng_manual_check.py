from pathlib import Path

import pandas as pd

from rec_tools.constants import DATA_AND_ARTIFACTS_DIR

feature_store_path = Path(DATA_AND_ARTIFACTS_DIR) / "feature_store"
splits_path = Path(DATA_AND_ARTIFACTS_DIR) / "train_val_test_splits"
temporal_splits_path = splits_path / "ts_movielens_splits"

# item dynamic features
ts_train_idf = pd.read_csv(feature_store_path / "ts_train_idf.csv")

# user dynamic features
ts_train_udf = pd.read_csv(feature_store_path / "ts_train_udf.csv")

# user-item interactions in the temporal split
ts_train_df = pd.read_csv(temporal_splits_path / "train.csv")

# pick 10 random users and check some features (make it deterministic)
random_users = ts_train_df["user_id"].sample(10).tolist()

for user in random_users:
    print(f"Checking user {user}")
    user_df = ts_train_df[ts_train_df["user_id"] == user]

    # number of ratings in original split
    num_ratings = len(user_df)

    # number of ratings in feature store
    num_ratings_fs = ts_train_udf[ts_train_udf["user_id"] == user][
        "total_ratings"
    ].values[0]

    assert num_ratings == num_ratings_fs

    # top 3 genres in original split
    top_genres = user_df["genres"].value_counts().head(3).index.tolist()
    # replace "|" with "_" and all to lowercase
    top_genres = sorted([genre.replace("|", "_").lower() for genre in top_genres])

    # top 3 genres in feature store
    top_genres_fs = sorted(
        ts_train_udf[ts_train_udf["user_id"] == user][
            ["favorite_genre_1", "favorite_genre_2", "favorite_genre_3"]
        ].values[0]
    )

    for i, genre in enumerate(top_genres):
        assert genre == top_genres_fs[i]


# pick 10 random items and check some features (make it deterministic)
random_items = ts_train_df["item_id"].sample(10, random_state=42).tolist()

for item in random_items:
    item_df = ts_train_df[ts_train_df["item_id"] == item]

    # unique users in original split
    nunique_users = item_df["user_id"].nunique()
    # unique users in feature store
    nunique_users_fs = ts_train_idf[ts_train_idf["item_id"] == item][
        "unique_reviewers"
    ].values[0]

    assert nunique_users == nunique_users_fs

    # occupation
    top_occupations = sorted(
        item_df["occupation"].value_counts().head(3).index.tolist()
    )
    occupation_columns = ["occupation_1", "occupation_2", "occupation_3"]
    top_occupations_fs = sorted(
        ts_train_idf[ts_train_idf["item_id"] == item][occupation_columns].values[0]
    )
    top_occupations_fs = [int(occupation) for occupation in top_occupations_fs]

    for i, occupation in enumerate(top_occupations):
        assert occupation == top_occupations_fs[i]

    # age
    top_ages = sorted(item_df["age"].value_counts().head(3).index.tolist())
    age_columns = ["age_1", "age_2", "age_3"]
    top_ages_fs = sorted(
        ts_train_idf[ts_train_idf["item_id"] == item][age_columns].values[0]
    )
    top_ages_fs = [int(age) for age in top_ages_fs]

    for i, age in enumerate(top_ages):
        assert age == top_ages_fs[i]

    # n male and female viewers
    n_male_viewers = item_df["gender"].value_counts().get("M", 0)
    n_female_viewers = item_df["gender"].value_counts().get("F", 0)
    n_male_viewers_fs = ts_train_idf[ts_train_idf["item_id"] == item][
        "male_viewers"
    ].values[0]
    n_female_viewers_fs = ts_train_idf[ts_train_idf["item_id"] == item][
        "female_viewers"
    ].values[0]

    assert n_male_viewers == n_male_viewers_fs
    assert n_female_viewers == n_female_viewers_fs
