import numpy as np
import pandas as pd


class ItemDynamicFeatures:

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all dynamic features for items."""

        dfc = df.copy()

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

        # Get demographic features from private method
        demographic_modes = self._compute_demographic_modes(dfc)

        gender_counts = dfc.groupby(["item_id", "gender"]).size().unstack(fill_value=0)

        features = pd.concat(  # type: ignore
            [rating_counts, rating_stats, demographic_modes, gender_counts],
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
            "age_1",
            "occupation_2",
            "age_2",
            "occupation_3",
            "age_3",
            "female_viewers",
            "male_viewers",
        ]

        features = features.reset_index()

        return features

    def _compute_demographic_modes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute demographic mode features for items based on user interactions."""
        demographic_modes = pd.DataFrame()

        for i, suffix in enumerate(["_1", "_2", "_3"]):
            rank_modes = (  # type: ignore
                df.groupby("item_id")
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

        return demographic_modes
