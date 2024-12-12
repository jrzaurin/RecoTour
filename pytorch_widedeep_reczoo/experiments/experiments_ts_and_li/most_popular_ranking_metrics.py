# We will simply load the full train dataset and the test set and, per each
# "group" of 100 items, we will calculate the hit ratio at k, map at k and ndcg
# at k. We will do this for both the ts and li datasets.

import pickle
from typing import Dict, List, Tuple
from pathlib import Path

import pandas as pd

from rec_tools.constants import (
    DATA_DIR,
    RESULTS_DIR,
    MOVIELENS_SPLITS_DIR,
    TRAIN_VAL_TEST_SPLITS_DIR,
)
from rec_tools.ranking_metrics import map_at_k, hit_ratio_at_k, binary_ndcg_at_k
from rec_tools.prepare_experiments.prepare_ts_or_li import binarize_target


def load_train_and_test_datasets() -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(
        Path(DATA_DIR)
        / TRAIN_VAL_TEST_SPLITS_DIR
        / MOVIELENS_SPLITS_DIR
        / "full_train.csv"
    )
    test_df = pd.read_csv(
        Path(DATA_DIR) / TRAIN_VAL_TEST_SPLITS_DIR / MOVIELENS_SPLITS_DIR / "test.csv"
    )

    return train_df, test_df


def item_popularity(
    train_df: pd.DataFrame, use_mean_rating: bool = False
) -> pd.DataFrame:
    items_popularity = (
        train_df.groupby("item_id")["rating"]
        .agg("mean" if use_mean_rating else "count")
        .reset_index()
    )
    items_popularity.rename(columns={"rating": "popularity"}, inplace=True)

    return items_popularity


def mp_ranking_metrics(
    use_mean_rating: bool = False,
    k_values: List[int] = [5, 10, 20],
) -> Dict[int, Dict[str, float]]:

    train_df, test_df = load_train_and_test_datasets()
    test_df = binarize_target(test_df)

    items_popularity = item_popularity(train_df, use_mean_rating)
    results_dir = (
        Path(RESULTS_DIR)
        / "results_most_popular_ranking_metrics"
        / f"results_mp_{'mean' if use_mean_rating else 'count'}"
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    test_df = test_df.merge(items_popularity, on="item_id", how="left")
    test_df["popularity"].fillna(0, inplace=True)

    y_test = test_df["rating"].values
    y_pred = test_df["popularity"].values

    results: Dict[int, Dict[str, float]] = {}
    for k in k_values:
        test_ndcg = binary_ndcg_at_k(y_pred, y_test, n_items=100, k=k)
        test_map = map_at_k(y_pred, y_test, n_items=100, k=k)
        test_hr = hit_ratio_at_k(y_pred, y_test, n_items=100, k=k)

        results[k] = {
            "ndcg": test_ndcg,
            "map": test_map,
            "hr": test_hr,
        }

        print(f"NDCG@{k}: {test_ndcg}")
        print(f"MAP@{k}: {test_map}")
        print(f"HR@{k}: {test_hr}")

    with open(results_dir / "results.pkl", "wb") as f:
        pickle.dump(results, f)

    return results


if __name__ == "__main__":
    mp_ranking_metrics(use_mean_rating=False)
    mp_ranking_metrics(use_mean_rating=True)
