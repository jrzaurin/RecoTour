import numpy as np
import pandas as pd

from rec_tools.feat_engineering.item_static_feats import process_genres


class UserDynamicFeatures:
    """Computes dynamic features for users based on their interactions."""

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all dynamic features for users."""

        dfc = df.copy()
        dfc = process_genres(dfc, "genres")

        # Basic count features
        viewing_counts = dfc.groupby("user_id").agg(
            {
                "rating": [
                    "count",
                    "mean",
                    "std",
                    "median",
                    lambda x: np.percentile(x, 75) - np.percentile(x, 25),
                ],  # rating stats
            }
        )

        # Genre-based features
        genre_features = self._compute_genre_preferences(dfc)

        # Combine all features
        features = pd.concat([viewing_counts, genre_features], axis=1)

        # Flatten column names and rename
        features.columns = [
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

        features = features.reset_index()

        return features

    def _compute_genre_preferences(self, merged_df: pd.DataFrame) -> pd.DataFrame:
        """Compute genre preferences for users based on view counts."""

        def get_top_3_genres_and_count(user_data):
            # Simply count the number of movies per genre
            genre_counts = user_data.groupby("genres").size()
            # Sort by count (descending)
            top_genres = genre_counts.sort_values(ascending=False).head(3)

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
                    "unique_genres_count": len(genre_counts),
                }
            )

        genre_preferences = merged_df.groupby("user_id").apply(
            get_top_3_genres_and_count
        )

        return genre_preferences
