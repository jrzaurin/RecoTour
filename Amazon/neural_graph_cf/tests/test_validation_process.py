import numpy as np

from multiprocessing import Pool
from dataset import Data

cores=4
n_users = 100
n_items = 1000
n_emb = 12
tr_items = 200
te_items = 200

np.random.seed(1)
training_items = np.random.choice(n_items, tr_items, replace=False)
all_items = np.arange(n_items)
test_items = np.setdiff1d(all_items, training_items)
user_pos_test = np.random.choice(test_items, te_items)
x = (np.random.choice(n_users, 1), np.random.rand(n_items))
Ks = [10,20]

# Commented out parts of the code that are not neccessary for this "dummy" test
import torch
from torch_metrics import ranklist_by_heapq, get_performance

emb_users = torch.from_numpy(np.random.rand(n_users, n_emb))
emb_items = torch.from_numpy(np.random.rand(n_items, n_emb))

users = np.arange(n_users)

def torch_test_one_user(x):
    u = x[0]
    rating = x[1]

    # try:
    #     training_items = data_generator.train_items[u]
    # except Exception:
    #     training_items = []

    # user_pos_test = data_generator.test_set[u]
    # all_items = set(range(data_generator.n_items))
    # test_items = list(all_items - set(training_items))

    r, auc = ranklist_by_heapq(user_pos_test, test_items, rating, Ks)

    return get_performance(user_pos_test, r, auc, Ks)


def torch_test():
	# Adapted the code for this "dummy" test

    result = {
        'precision': np.zeros(len(Ks)),
        'recall': np.zeros(len(Ks)),
        'ndcg': np.zeros(len(Ks)),
        'hit_ratio': np.zeros(len(Ks)),
        'auc': 0.
        }
    p = Pool(cores)
    user_batch = users
    user_emb = emb_users
    item_emb = emb_items
    rate_batch  = torch.mm(user_emb, item_emb.t())
    rate_batch = rate_batch.cpu().numpy()
    batch_result = p.map(torch_test_one_user, zip(user_batch,rate_batch))

    for re in batch_result:
        result['precision'] += re['precision']/n_users
        result['recall'] += re['recall']/n_users
        result['ndcg'] += re['ndcg']/n_users
        result['hit_ratio'] += re['hit_ratio']/n_users
        result['auc'] += re['auc']/n_users

    p.close()
    return result

torch_res_one_user = torch_test_one_user(x)
torch_res = torch_test()

#TF
import tensorflow as tf
from tf_metrics import ranklist_by_heapq, get_performance

tf.enable_eager_execution()

emb_users = tf.convert_to_tensor(emb_users.numpy())
emb_items = tf.convert_to_tensor(emb_items.numpy())

users = np.arange(n_users)

def tf_test_one_user(x):
    # user u's ratings for user u
    rating = x[1]
    #uid
    u = x[0]
    #user u's items in the training set
    # try:
    #     training_items = data_generator.train_items[u]
    # except Exception:
    #     training_items = []
    # #user u's items in the test set
    # user_pos_test = data_generator.test_set[u]

    # all_items = set(range(ITEM_NUM))

    # test_items = list(all_items - set(training_items))

    r, auc = ranklist_by_heapq(user_pos_test, test_items, rating, Ks)

    return get_performance(user_pos_test, r, auc, Ks)


def tf_test():
    result = {'precision': np.zeros(len(Ks)), 'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks)),
              'hit_ratio': np.zeros(len(Ks)), 'auc': 0.}

    p = Pool(cores)
    test_users = users
    n_test_users = len(test_users)
    user_batch = users
    user_emb = emb_users
    item_emb = emb_items

    rate_batch = tf.matmul(user_emb, item_emb, transpose_a=False, transpose_b=True)
    user_batch_rating_uid = zip(user_batch, rate_batch)
    batch_result = p.map(tf_test_one_user, user_batch_rating_uid)

    for re in batch_result:
        result['precision'] += re['precision']/n_test_users
        result['recall'] += re['recall']/n_test_users
        result['ndcg'] += re['ndcg']/n_test_users
        result['hit_ratio'] += re['hit_ratio']/n_test_users
        result['auc'] += re['auc']/n_test_users
    p.close()
    return result

tf_res_one_user = tf_test_one_user(x)
tf_res = tf_test()

