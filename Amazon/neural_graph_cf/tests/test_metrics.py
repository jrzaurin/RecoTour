'''
Test that the metrics I implemented lead to identical results to that of the
original code release. In this dir tf_metrics.py is a direct copy and paste
from here: https://github.com/xiangwang1223/neural_graph_collaborative_filtering/blob/master/NGCF/utility/metrics.py
'''
import numpy as np

np.random.seed(1)
r = np.random.choice(2, 100, p=[0.6, 0.4])
n_inter = 50
k = 10

# RECALL@K
from tf_metrics import recall_at_k as tf_recall_at_k
from torch_metrics import recall_at_k as torch_recall_at_k

tf_rec = tf_recall_at_k(r, k, n_inter)
torch_rec = torch_recall_at_k(r, k, n_inter)
print(tf_rec, torch_rec)

# PRECISION@K
from tf_metrics import precision_at_k as tf_precision_at_k
from torch_metrics import precision_at_k as torch_precision_at_k

tf_prec = tf_precision_at_k(r, k)
torch_prec = torch_precision_at_k(r, k)
print(tf_prec, torch_prec)

# NDCG@K
from tf_metrics import ndcg_at_k as tf_ndcg_at_k
from torch_metrics import ndcg_at_k as torch_ndcg_at_k

tf_prec = tf_ndcg_at_k(r, k)
torch_prec = torch_ndcg_at_k(r, k)
print(tf_prec, torch_prec)

# HIT@K
from tf_metrics import hit_at_k as tf_hit_at_k
from torch_metrics import hit_at_k as torch_hit_at_k

tf_prec = tf_hit_at_k(r, k)
torch_prec = torch_hit_at_k(r, k)
print(tf_prec, torch_prec)

# ALL METRICS
np.random.seed(1)
item_score = dict(zip(np.arange(1000), np.random.rand(1000)))
test_items = list(item_score.keys())
rating = list(item_score.values())
user_pos_test = np.random.choice(1000, 100, replace=False)

from tf_metrics import ranklist_by_heapq as tf_ranklist_by_heapq
from tf_metrics import get_performance as tf_get_performance
from torch_metrics import ranklist_by_heapq as torch_ranklist_by_heapq
from torch_metrics import get_performance as torch_get_performance

tf_rank, auc = tf_ranklist_by_heapq(user_pos_test, test_items, rating, [10, 20])
torch_rank, auc = torch_ranklist_by_heapq(user_pos_test, test_items, rating, [10, 20])
print(tf_rank)
print(torch_rank)

tf_score = tf_get_performance(user_pos_test, tf_rank, auc, [10, 20])
torch_score = torch_get_performance(user_pos_test, tf_rank, auc, [10, 20])
print(tf_score)
print(torch_score)

