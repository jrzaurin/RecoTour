'''
This code is mostly taken from here:
https://github.com/xiangwang1223/neural_graph_collaborative_filtering

This is part of my Pytorch adaptation of:
Wang Xiang et al. Neural Graph Collaborative Filtering. In SIGIR 2019.
'''
import numpy as np
import heapq
from sklearn.metrics import roc_auc_score


def recall_at_k(r, k, all_pos_num):
    """recall @ k
    Parameters:
    ----------
    r: Iterable
        binary iterable (nonzero is relevant).
    k: Int
        number of recommendations to consider
    all_pos_num: Int
        number of interactions
    Returns:
    ----------
    recall @ k
    """
    r = np.asfarray(r)[:k]
    return np.sum(r) / all_pos_num


def precision_at_k(r, k):
    """
    Precision@K

    Parameters:
    ----------
    r: Iterable
        Binary iterable where nonzero is relevant
    k: Int

    Returns:
    Precision @ k
    -------
    """
    assert k >= 1
    r = np.asarray(r)[:k]
    return np.mean(r)


def dcg_at_k(r, k, method=1):
    """
    discounted cumulative gain (dcg@k) at k

    Parameters:
    ----------
    r: Iterable
        Binary iterable where nonzero is relevant
    k: Int
    method: Int
        Indicates the DCG expression to use

    Returns:
    DCG@k
    """
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def ndcg_at_k(r, k, method=1):
    """
    Normalized discounted cumulative gain (NDGC@K) at k
    """
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max


def hit_at_k(r, k):
    """hit ratio @ k
    Parameters:
    ----------
    r: Iterable
        binary iterable (nonzero is relevant).
    k: Int
        number of recommendations to consider
    Returns:
    ----------
    hit ratio @ k
    """
    r = np.array(r)[:k]
    if np.sum(r) > 0:
        return 1.
    else:
        return 0.


def ranklist_by_heapq(user_pos_test, test_items, rating, Ks):
    """
    Retursn a binary list, where relevance is nonzero, based on a ranked list
    with the n largest scores.
    Parameters:
    ----------
    user_pos_test: List
        List with the items that the user actually interacted with
    test_items: List
        List with the all items in the test dataset
    rating: List
        List with the ratings corresponding to test_items
    Ks: Int or List
        the k in @k
    Returns:
    ----------
    r: binary list where nonzero in relevant
    """
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]

    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    return r


def get_performance(user_pos_test, r, Ks):
    """wrap up around all other previous functions
    ----------
    user_pos_test: List
        List with the items that the user actually interacted with
    r: List
        binary list where nonzero in relevant
    Ks: List
        the k in @k
    Returns:
    ----------
    dictionary of metrics
    """
    precision, recall, ndcg, hit_ratio = [], [], [], []

    for K in Ks:
        precision.append(precision_at_k(r, K))
        recall.append(recall_at_k(r, K, len(user_pos_test)))
        ndcg.append(ndcg_at_k(r, K))
        hit_ratio.append(hit_at_k(r, K))

    return {'recall': np.array(recall), 'precision': np.array(precision),
            'ndcg': np.array(ndcg), 'hit_ratio': np.array(hit_ratio)}
