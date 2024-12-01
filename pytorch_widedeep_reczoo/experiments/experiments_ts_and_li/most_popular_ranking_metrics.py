# We will simply load the full train dataset and the test set and, per each
# "group" of 100 items, we will calculate the hit ratio at k, map at k and ndcg
# at k. We will do this for both the ts and li datasets.

import pickle
from typing import Dict, List, Tuple
from pathlib import Path
from functools import partial
from multiprocessing import Pool, cpu_count

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


def process_single_user(
    user_id: int,
    *,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    use_mean_rating: bool,
) -> pd.DataFrame:
    user_test_items = test_df[test_df["user_id"] == user_id]
    items_in_train = train_df[train_df["item_id"].isin(user_test_items["item_id"])]

    items_popularity = (
        items_in_train.groupby("item_id")["rating"]
        .agg("mean" if use_mean_rating else "count")
        .reset_index()
    )
    items_popularity.rename(columns={"rating": "popularity"}, inplace=True)

    return user_test_items.merge(items_popularity, on="item_id", how="left")


def most_popular_predictions(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    use_mean_rating: bool = False,
    n_cpus: int | None = None,
) -> pd.DataFrame:
    process_user = partial(
        process_single_user,
        train_df=train_df,
        test_df=test_df,
        use_mean_rating=use_mean_rating,
    )

    n_cpus = n_cpus or cpu_count()
    with Pool(n_cpus) as pool:
        predictions = pool.map(process_user, test_df["user_id"].unique())

    final_df = pd.concat(predictions, ignore_index=True)
    final_df = binarize_target(final_df)

    return final_df[["user_id", "item_id", "rating", "popularity"]]


def mp_ranking_metrics(
    use_mean_rating: bool = False,
    k_values: List[int] = [5, 10, 20],
) -> Dict[int, Dict[str, float]]:

    train_df, test_df = load_train_and_test_datasets()
    predictions = most_popular_predictions(train_df, test_df, use_mean_rating)

    results_dir = Path(RESULTS_DIR) / "results_most_popular_ranking_metrics"
    results_dir.mkdir(parents=True, exist_ok=True)

    y_test = predictions["rating"].values
    y_pred = predictions["popularity"].values

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
    mp_ranking_metrics()
