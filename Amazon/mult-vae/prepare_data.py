import os
import pickle
import sys
from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np
import pandas as pd
from fire import Fire


def get_count(tp: pd.DataFrame, id: str) -> pd.Index:
    playcount_groupbyid = tp[[id]].groupby(id, as_index=False)
    count = playcount_groupbyid.size()
    return count


def filter_triplets(
    tp: pd.DataFrame, min_user_click, min_item_click
) -> Tuple[pd.DataFrame, pd.Index, pd.Index]:

    if min_item_click > 0:
        itemcount = get_count(tp, "item")
        tp = tp[tp["item"].isin(itemcount.index[itemcount >= min_item_click])]

    if min_user_click > 0:
        usercount = get_count(tp, "user")
        tp = tp[tp["user"].isin(usercount.index[usercount >= min_user_click])]

    usercount, itemcount = get_count(tp, "user"), get_count(tp, "item")

    return tp, usercount, itemcount


def split_users(
    unique_uid: pd.Index, test_users_size: Union[float, int]
) -> Tuple[pd.Index, pd.Index, pd.Index]:

    n_users = unique_uid.size

    if isinstance(test_users_size, int):
        n_heldout_users = test_users_size
    else:
        n_heldout_users = int(test_users_size * n_users)

    tr_users = unique_uid[: (n_users - n_heldout_users * 2)]
    vd_users = unique_uid[(n_users - n_heldout_users * 2) : (n_users - n_heldout_users)]
    te_users = unique_uid[(n_users - n_heldout_users) :]

    return tr_users, vd_users, te_users


def split_train_test(
    data: pd.DataFrame, test_size: float
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    data_grouped_by_user = data.groupby("user")
    tr_list, te_list = list(), list()

    np.random.seed(98765)

    for i, (nm, group) in enumerate(data_grouped_by_user):
        n_items_u = len(group)

        if n_items_u >= 5:
            idx = np.zeros(n_items_u, dtype="bool")
            idx[
                np.random.choice(
                    n_items_u, size=int(test_size * n_items_u), replace=False
                ).astype("int64")
            ] = True

            tr_list.append(group[np.logical_not(idx)])
            te_list.append(group[idx])
        else:
            tr_list.append(group)

        if i % 1000 == 0:
            print("%d users sampled" % i)
            sys.stdout.flush()

    data_tr = pd.concat(tr_list)
    data_te = pd.concat(te_list)

    return data_tr, data_te


def numerize(tp: pd.DataFrame, user2id: Dict, item2id: Dict) -> pd.DataFrame:
    user = [user2id[x] for x in tp["user"]]
    item = [item2id[x] for x in tp["item"]]
    return pd.DataFrame(data={"user": user, "item": item}, columns=["user", "item"])


def main(
    dataset: str,
    min_user_click=5,
    min_item_click=0,
    test_users_size: Union[float, int] = 0.1,
    test_size=0.2,
):

    DATA_DIR = Path("data")
    new_colnames = ["user", "item", "rating", "timestamp"]
    if dataset == "amazon":
        inp_path = DATA_DIR / "amazon-movies"
        filename = "reviews_Movies_and_TV_5.json.gz"
        # filename = "reviews_Movies_and_TV_5.p"

        raw_data = pd.read_json(inp_path / filename, lines=True)
        # raw_data = pd.read_pickle(inp_path / filename)
        keep_cols = ["reviewerID", "asin", "overall", "unixReviewTime"]
        raw_data = raw_data[keep_cols]
        raw_data.columns = new_colnames
        # raw_data = raw_data[raw_data["rating"] > 3]

    elif dataset == "movielens":
        inp_path = DATA_DIR / "ml-20m"
        filename = "ratings.csv"

        raw_data = pd.read_csv(inp_path / filename, header=0)
        raw_data.columns = new_colnames
        raw_data = raw_data[raw_data["rating"] > 3.5]

    filtered_raw_data, user_activity, item_popularity = filter_triplets(
        raw_data, min_user_click=min_user_click, min_item_click=min_item_click
    )
    sparsity = (
        1.0
        * filtered_raw_data.shape[0]
        / (user_activity.shape[0] * item_popularity.shape[0])
    )

    print(
        "After filtering, there are %d watching events from %d users and %d movies (sparsity: %.3f%%)"
        % (
            filtered_raw_data.shape[0],
            user_activity.shape[0],
            item_popularity.shape[0],
            sparsity * 100,
        )
    )

    unique_uid = user_activity.index
    np.random.seed(98765)
    idx_perm = np.random.permutation(unique_uid.size)
    unique_uid = unique_uid[idx_perm]
    tr_users, vd_users, te_users = split_users(
        unique_uid, test_users_size=test_users_size
    )

    tr_obsrv = filtered_raw_data.loc[filtered_raw_data["user"].isin(tr_users)]
    tr_items = pd.unique(tr_obsrv["item"])

    item2id = dict((sid, i) for (i, sid) in enumerate(tr_items))
    user2id = dict((pid, i) for (i, pid) in enumerate(unique_uid))

    out_path = DATA_DIR / "_".join([dataset, "processed"])
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    np.save(out_path / "tr_items", tr_items)
    pickle.dump(item2id, open(out_path / "item2id.p", "wb"))
    pickle.dump(user2id, open(out_path / "user2id.p", "wb"))

    vd_obsrv = filtered_raw_data[
        filtered_raw_data["user"].isin(vd_users)
        & filtered_raw_data["item"].isin(tr_items)
    ]
    vd_items_tr, vd_items_te = split_train_test(vd_obsrv, test_size=test_size)

    te_obsrv = filtered_raw_data[
        filtered_raw_data["user"].isin(te_users)
        & filtered_raw_data["item"].isin(tr_items)
    ]
    te_items_tr, te_items_te = split_train_test(te_obsrv, test_size=test_size)

    tr_data = numerize(tr_obsrv, user2id, item2id)
    tr_data.to_csv(out_path / "train.csv", index=False)

    vd_data_tr = numerize(vd_items_tr, user2id, item2id)
    vd_data_tr.to_csv(out_path / "validation_tr.csv", index=False)

    vd_data_te = numerize(vd_items_te, user2id, item2id)
    vd_data_te.to_csv(out_path / "validation_te.csv", index=False)

    te_data_tr = numerize(te_items_tr, user2id, item2id)
    te_data_tr.to_csv(out_path / "test_tr.csv", index=False)

    te_data_te = numerize(te_items_te, user2id, item2id)
    te_data_te.to_csv(out_path / "test_te.csv", index=False)


if __name__ == "__main__":
    Fire(main)
