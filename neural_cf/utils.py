import numpy as np
import pandas as pd
import math
import heapq


def get_train_instances(train, negatives, n_items, n_neg):
    user, item, labels = [],[],[]
    for (u, i), r in train.items():
        # positive instance
        user.append(u)
        item.append(i)
        labels.append(1)
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


def get_hitratio(ranklist, gtitem):
    if gtitem in ranklist: return 1
    return 0


def get_ndcg(ranklist, gtitem):
    for i in range(len(ranklist)):
        item = ranklist[i]
        if item == gtitem:
            return math.log(2) / math.log(i+2)
    return 0


def get_scores(items, preds, topk):

    gtitem = items[0]

    # the following 3 lines of code ensure that the fact that the 1st item is
    # gtitem does not affect the final rank
    randidx = np.arange(100)
    np.random.shuffle(randidx)
    items, preds = items[randidx], preds[randidx]

    map_item_score = dict( zip(items, preds) )
    ranklist = heapq.nlargest(topk, map_item_score, key=map_item_score.get)
    hr = get_hitratio(ranklist, gtitem)
    ndcg = get_ndcg(ranklist, gtitem)
    return hr, ndcg
