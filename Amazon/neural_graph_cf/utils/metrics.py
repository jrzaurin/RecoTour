import numpy as np
import heapq

from sklearn.metrics import roc_auc_score


def recall_at_k(r, k, n_inter):
    """recall @ k
    Parameters:
    ----------
    r: Int
        binary iterable (nonzero is relevant).
    k: Int
        number of recommendations to consider
    n_inter: Int
        number of interactions
    Returns:
    ----------
    recall @ k
    """
    r = np.asfarray(r)[:k]
    return np.sum(r) / n_inter


def precision_at_k(r, k):
    """precision @ k
    Parameters:
    ----------
    r: Int
        binary iterable (nonzero is relevant).
    k: Int
        number of recommendations to consider
    Returns:
    ----------
    Precision @ k
    """
    assert k >= 1
    r = np.asarray(r)[:k]
    return np.mean(r)


def dcg_at_k(r, k, method=1):
    """ discounted cumulative gain (dcg) @ k
    Parameters:
    ----------
    r: Int or Float
        Relevance is positive real values. If binary, nonzero is relevant.
    k: Int
        number of recommendations to consider
    method: Int
        one of 0 or 1. Simply, different dcg implementations
    Returns:
    ----------
    dcg @ k
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
    """ Normalized discounted cumulative gain @ k
    """
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max


def hit_at_k(r, k):
    """hit ratio @ k
    Parameters:
    ----------
    r: Int
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


def auc(true, pred):
    """Simple wrap up around sklearn's roc_auc_score
    """
    try:
        res = roc_auc_score(true, pred)
    except Exception:
        res = 0.
    return res


def get_auc(item_score, user_pos_test):
    """Wrap up around sklearn's roc_auc_score
    Parameters:
    ----------
    item_score: Dict
        Dict. keys are item_ids, values are predictions
    user_pos_test: List
        List with the items that the user actually interacted with
    Returns:
    ----------
    res: Float
        roc_auc_score
    """
    item_score = sorted(item_score.items(), key=lambda kv: kv[1])
    item_score.reverse()
    item_id = [x[0] for x in item_score]
    score = [x[1] for x in item_score]

    r = []
    for i in item_id:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)

    try:
        res = roc_auc_score(r, score)
    except Exception:
        res = 0.

    return res


def ranklist_by_heapq(user_pos_test, test_items, rating, Ks):
    """
    Retursn a binary list, where relevance is nonzero, based on a ranked list
    with the n largest scores. For consistency with ranklist_by_sorted, also
    returns auc=0 (since auc does not make sense within a mini batch)
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
    auc = 0.
    return r, auc


def ranklist_by_sorted(user_pos_test, test_items, rating, Ks):
    """
    Retursn a binary list, where relevance is nonzero, based on a ranked list
    with the n largest scores. Also returns the AUC
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
    auc: testing roc_auc_score
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
    auc = get_auc(item_score, user_pos_test)
    return r, auc


def get_performance(user_pos_test, r, auc, Ks):
    """wrap up around all other previous functions
    ----------
    user_pos_test: List
        List with the items that the user actually interacted with
    r: List
        binary list where nonzero in relevant
    auc: Float
        sklearn's roc_auc_score
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
            'ndcg': np.array(ndcg), 'hit_ratio': np.array(hit_ratio), 'auc': auc}
