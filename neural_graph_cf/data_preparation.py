"""
Data preparation process. I will use two methods. One designed to reproduced
Xiang Wang Neural Graph Collaborative Filtering paper and the other one using
an approach based on Xiangnan He Neural Collaborative Filtering paper
"""
import numpy as np
import pandas as pd
import pickle
import os
import scipy.sparse as sp
import argparse
import csv

from copy import copy
from time import time
from tqdm import tqdm
from pathlib import Path
from joblib import Parallel, delayed
from scipy.sparse import save_npz


def map_user_items(df):
    """
    Function to map users and items into continuous integers
    """
    dfc = df.copy()
    user_mappings = {k: v for v, k in enumerate(dfc.user.unique())}
    item_mappings = {k: v for v, k in enumerate(dfc.item.unique())}

    # save with " " separated format
    user_list = pd.DataFrame.from_dict(user_mappings, orient="index").reset_index()
    user_list.columns = ["orig_id", "remap_id"]
    item_list = pd.DataFrame.from_dict(item_mappings, orient="index").reset_index()
    item_list.columns = ["orig_id", "remap_id"]
    user_list.to_csv(DATA_PATH / "user_list.txt", sep=" ", index=False)
    item_list.to_csv(DATA_PATH / "item_list.txt", sep=" ", index=False)

    dfc["user"] = dfc["user"].map(user_mappings).astype(np.int64)
    dfc["item"] = dfc["item"].map(item_mappings).astype(np.int64)

    return user_mappings, item_mappings, dfc


def tolist(df):
    """
    Build a dataframe (user, list of items)
    """
    keys, values = df.sort_values("user").values.T
    ukeys, index = np.unique(keys, True)
    arrays = np.split(values, index[1:])
    df2 = pd.DataFrame({"user": ukeys, "item": [list(a) for a in arrays]})
    return df2


def train_test_split(u, i_l, p=0.8):
    s = np.floor(len(i_l) * p).astype("int")
    train = list(np.random.choice(i_l, s, replace=False))
    test = list(np.setdiff1d(i_l, train))
    return ([u] + train, [u] + test)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Prepare the Amazon movies dataset.")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/home/ubuntu/projects/RecoTour/datasets/Amazon",
        help="Dir path for raw data",
    )
    parser.add_argument(
        "--input_data",
        type=str,
        default="reviews_Movies_and_TV_5.json.gz",
        help="File name for raw data",
    )
    parser.add_argument(
        "--valid", type=int, default=1, help="train/valid/test or train/test split"
    )
    args = parser.parse_args()

    DATA_PATH = Path(args.input_dir)
    reviews = args.input_data
    reviews_csv = "reviews_Movies_and_TV_5.csv"

    print("Reading amazon movies dataset...")
    df = pd.read_json(DATA_PATH / reviews, lines=True)
    keep_cols = ["reviewerID", "asin", "unixReviewTime", "overall"]
    new_colnames = ["user", "item", "timestamp", "rating"]
    df = df[keep_cols]
    df.columns = new_colnames
    df.to_csv(DATA_PATH / reviews_csv, index=False)

    df.sort_values(["user", "timestamp"], ascending=[True, True], inplace=True)
    df.reset_index(inplace=True, drop=True)

    user_mappings, item_mappings, dfm = map_user_items(df)

    print("creating dataframe with lists of interactions...")
    df1 = dfm[["user", "item"]]
    interactions_df = tolist(df1)

    # train/test split
    print("Train/Test split (and save)...")
    interactions_l = [
        train_test_split(r["user"], r["item"]) for i, r in interactions_df.iterrows()
    ]
    train = [interactions_l[i][0] for i in range(len(interactions_l))]
    test = [interactions_l[i][1] for i in range(len(interactions_l))]
    train_fname = DATA_PATH / "train.txt"
    test_fname = DATA_PATH / "test.txt"

    if args.valid:
        # train/valid split
        tr_interactions_l = [train_test_split(t[0], t[1:], p=0.8) for t in train]
        train = [tr_interactions_l[i][0] for i in range(len(tr_interactions_l))]
        valid = [tr_interactions_l[i][1] for i in range(len(tr_interactions_l))]
        valid_fname = DATA_PATH / "valid.txt"
        with open(train_fname, "w") as trf, open(valid_fname, "w") as vaf, open(
            test_fname, "w"
        ) as tef:
            trwrt = csv.writer(trf, delimiter=" ")
            vawrt = csv.writer(vaf, delimiter=" ")
            tewrt = csv.writer(tef, delimiter=" ")
            trwrt.writerows(train)
            vawrt.writerows(valid)
            tewrt.writerows(test)
    else:
        with open(train_fname, "w") as trf, open(test_fname, "w") as tef:
            trwrt = csv.writer(trf, delimiter=" ")
            tewrt = csv.writer(tef, delimiter=" ")
            trwrt.writerows(train)
            tewrt.writerows(test)
