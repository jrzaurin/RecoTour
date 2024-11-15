import numpy as np
import pandas as pd


class ItemDynamicFeatures:

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all dynamic features for items."""

        dfc = df.copy()
        dfc["timestamp"] = pd.to_datetime(dfc["timestamp"])

        rating_counts = dfc.groupby("item_id").agg(
            {"rating": ["count", "nunique"], "user_id": "nunique"}
        )

        rating_stats = dfc.groupby("item_id").agg(
            {
                "rating": [
                    "median",
                    "mean",
                    "std",
                    lambda x: np.percentile(x, 75) - np.percentile(x, 25),
                ]
            }
        )

        # Create separate aggregations for each rank
        demographic_modes = pd.DataFrame()

        for i, suffix in enumerate(["_1", "_2", "_3"]):
            rank_modes = (
                dfc.groupby("item_id")
                .agg(
                    {
                        "occupation": lambda x: (
                            x.astype(str).value_counts().nlargest(3).index[i]
                            if len(x.value_counts()) > i
                            else None
                        ),
                        "age": lambda x: (
                            x.astype(str).value_counts().nlargest(3).index[i]
                            if len(x.value_counts()) > i
                            else None
                        ),
                    }
                )
                .rename(
                    columns={"occupation": f"occupation{suffix}", "age": f"age{suffix}"}
                )
            )

            demographic_modes = pd.concat([demographic_modes, rank_modes], axis=1)

        gender_counts = dfc.groupby(["item_id", "gender"]).size().unstack(fill_value=0)

        time_features = self._compute_time_features(dfc)

        features = pd.concat(
            [  # type: ignore
                rating_counts,
                rating_stats,
                demographic_modes,
                gender_counts,
                time_features,
            ],
            axis=1,
        )

        # Flatten column names and rename
        features.columns = [
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

        features = features.reset_index()

        return features

    @staticmethod
    def _compute_time_features(df: pd.DataFrame) -> pd.DataFrame:
        """Compute time-based features for items."""

        time_features = []

        for item_id in df["item_id"].unique():
            item_ratings = df[df["item_id"] == item_id]["timestamp"].sort_values()

            if len(item_ratings) > 1:
                # Calculate mean days between ratings
                days_between = item_ratings.diff().mean().total_seconds() / (24 * 3600)

                # Calculate recency (days since last rating)
                last_rating = item_ratings.max()
                recency = (df["timestamp"].max() - last_rating).total_seconds() / (
                    24 * 3600
                )

                # Calculate timespan of ratings
                timespan = (item_ratings.max() - item_ratings.min()).total_seconds() / (
                    24 * 3600
                )
            else:
                days_between = 0
                recency = 0
                timespan = 0

            time_features.append(
                {
                    "item_id": item_id,
                    "mean_days_between_ratings": days_between,
                    "rating_recency_days": recency,
                    "rating_timespan_days": timespan,
                }
            )

        return pd.DataFrame(time_features).set_index("item_id")
