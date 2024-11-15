import numpy as np
import pandas as pd

from rec_zoo.feat_engineering.utils import save_objects
from rec_zoo.feat_engineering.item_static_feats import process_genres


class UserDynamicFeatures:
    """Computes dynamic features for users based on their interactions."""

    def __init__(self, save_dir: str | None = "feature_store"):

        self.save_dir = save_dir

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all dynamic features for users."""

        dfc = df.copy()
        dfc["timestamp"] = pd.to_datetime(dfc["timestamp"])
        dfc = process_genres(dfc, "genres")

        # Basic count features
        viewing_counts = dfc.groupby("user_id").agg(
            {
                "item_id": ["count", "nunique"],  # total and unique movies
                "rating": [
                    "count",
                    "mean",
                    "std",
                    "median",
                    lambda x: np.percentile(x, 75) - np.percentile(x, 25),
                ],  # rating stats
            }
        )

        # Time-based features
        time_features = self._compute_time_features(dfc)

        # Genre-based features
        genre_features = self._compute_genre_preferences(dfc)

        # Combine all features
        features = pd.concat([viewing_counts, time_features, genre_features], axis=1)

        # Flatten column names and rename
        features.columns = [
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

        features = features.reset_index()

        if self.save_dir is not None:
            save_objects([features], ["user_dynamic_features.csv"], self.save_dir)

        return features

    def _compute_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute time-based features for users."""

        time_features = []

        for user_id in df["user_id"].unique():
            user_views = df[df["user_id"] == user_id]["timestamp"].sort_values()

            if len(user_views) > 1:
                # Calculate mean days between views
                days_between = user_views.diff().mean().total_seconds() / (24 * 3600)

                # Calculate recency (days since last view)
                last_view = user_views.max()
                recency = (df["timestamp"].max() - last_view).total_seconds() / (
                    24 * 3600
                )

                # Calculate timespan of views
                timespan = (user_views.max() - user_views.min()).total_seconds() / (
                    24 * 3600
                )

                # Calculate weekly viewing frequency
                weeks = timespan / 7
                weekly_frequency = len(user_views) / weeks if weeks > 0 else 0
            else:
                days_between = 0
                recency = 0
                timespan = 0
                weekly_frequency = 0

            time_features.append(
                {
                    "user_id": user_id,
                    "mean_days_between_views": days_between,
                    "viewing_recency_days": recency,
                    "viewing_timespan_days": timespan,
                    "viewing_frequency_weekly": weekly_frequency,
                }
            )

        return pd.DataFrame(time_features).set_index("user_id")

    def _compute_genre_preferences(self, merged_df: pd.DataFrame) -> pd.DataFrame:
        """Compute genre preferences for users based on average ratings."""

        def get_top_3_genres_and_count(user_data):
            genre_ratings = user_data.groupby("genres")["rating"].agg(["mean", "count"])
            # Sort by mean rating (descending), then by count (descending) for tiebreaking
            top_genres = genre_ratings.sort_values(
                ["mean", "count"], ascending=[False, False]
            ).head(3)

            return pd.Series(
                {
                    "favorite_genre_1": (
                        top_genres.index[0] if len(top_genres) > 0 else np.nan
                    ),
                    "favorite_genre_2": (
                        top_genres.index[1] if len(top_genres) > 1 else np.nan
                    ),
                    "favorite_genre_3": (
                        top_genres.index[2] if len(top_genres) > 2 else np.nan
                    ),
                    "unique_genres_count": len(genre_ratings),
                }
            )

        genre_preferences = merged_df.groupby("user_id").apply(
            get_top_3_genres_and_count
        )

        return genre_preferences
