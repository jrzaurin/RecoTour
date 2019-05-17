import numpy as np
import pandas as pd
import math
import pdb

from scipy.sparse import load_npz
from tqdm import tqdm


def get_train_instances(train, negatives, n_items, n_neg, binary=True):
    user, item, labels = [],[],[]
    for (u, i), r in tqdm(train.items(), total=len(train)):
        # positive instance
        user.append(u)
        item.append(i)
        if binary:
            labels.append(1)
        else:
            labels.append(r)
        # negative instances: we also need to make sure they are not in the
        # negative examples used for testing
        for _ in range(n_neg):
            j = np.random.randint(n_items)
            while ((u, j) in train.keys()) or (j in negatives[u]):
                j = np.random.randint(n_items)
            user.append(u)
            item.append(j)
            labels.append(0)
    train_w_negative = np.vstack([user,item,labels]).T
    assert train_w_negative.shape[0] == (len(train) + len(train)*n_neg)
    return train_w_negative.astype(np.int64)


def get_hitratio(ranklist, gtItem):
    for item in ranklist:
        if item == gtItem:
            return 1
    return 0


def get_ndcg(ranklist, gtItem):
    for i in range(len(ranklist)):
        item = ranklist[i]
        if item == gtItem:
            return math.log(2) / math.log(i+2)
    return 0
