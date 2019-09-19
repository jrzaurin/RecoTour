import numpy as np
import pandas as pd
import os
import pickle

from time import time
from sklearn.metrics.pairwise import pairwise_distances
from scipy.sparse import csr_matrix, load_npz
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from recutils.average_precision import mapk
from multiprocessing import Pool

import multiprocessing
from joblib import Parallel, delayed

inp_dir = "../../datasets/Ponpare/data_processed/"
train_dir = "train"
valid_dir = "valid"
# train and validation coupons
df_coupons_train_feat = pd.read_pickle(os.path.join(inp_dir, train_dir, 'df_coupons_train_feat.p'))
df_coupons_valid_feat = pd.read_pickle(os.path.join(inp_dir, valid_dir, 'df_coupons_valid_feat.p'))
coupons_train_ids = df_coupons_train_feat.coupon_id_hash.values
coupons_valid_ids = df_coupons_valid_feat.coupon_id_hash.values

# let's add a flag for convenience
df_coupons_train_feat['flag_cat'] = 0
df_coupons_valid_feat['flag_cat'] = 1

flag_cols = ['flag_cat_0','flag_cat_1']

cat_cols = [c for c in df_coupons_train_feat.columns if '_cat' in c]
id_cols = ['coupon_id_hash']
num_cols = [c for c in df_coupons_train_feat.columns if
	(c not in cat_cols) and (c not in id_cols)]

tmp_df = pd.concat([df_coupons_train_feat[cat_cols],
	df_coupons_valid_feat[cat_cols]],
	ignore_index=True)

df_dummy_feats = pd.get_dummies(tmp_df.astype('category'))

coupons_train_feat_oh = (df_dummy_feats[df_dummy_feats.flag_cat_0 != 0]
	.drop(flag_cols, axis=1)
	.values)
coupons_valid_feat_oh = (df_dummy_feats[df_dummy_feats.flag_cat_1 != 0]
	.drop(flag_cols, axis=1)
	.values)

coupons_train_feat_num = df_coupons_train_feat[num_cols].values
coupons_valid_feat_num = df_coupons_valid_feat[num_cols].values

scaler = MinMaxScaler()
coupons_train_feat_num_norm = scaler.fit_transform(coupons_train_feat_num)
coupons_valid_feat_num_norm = scaler.transform(coupons_valid_feat_num)

coupons_train_feat = np.hstack([coupons_train_feat_num_norm, coupons_train_feat_oh])
coupons_valid_feat = np.hstack([coupons_valid_feat_num_norm, coupons_valid_feat_oh])

dist_mtx = pairwise_distances(coupons_valid_feat, coupons_train_feat, metric='cosine')

# now we have a matrix of distances, let's build the dictionaries
valid_to_train_top_n_idx = np.apply_along_axis(np.argsort, 1, dist_mtx)
train_to_valid_top_n_idx = np.apply_along_axis(np.argsort, 1, dist_mtx.T)
train_to_valid_most_similar = dict(zip(coupons_train_ids,
	coupons_valid_ids[train_to_valid_top_n_idx[:,0]]))
# there is one coupon in validation: '0a8e967835e2c20ac4ed8e69ee3d7349' that
# is never among the most similar to those previously seen.
valid_to_train_most_similar = dict(zip(coupons_valid_ids,
	coupons_train_ids[valid_to_train_top_n_idx[:,0]]))

# build a dictionary or interactions during training
df_interest = pd.read_pickle(os.path.join(inp_dir, train_dir, "df_interest.p"))
df_interactions_train = (df_interest.groupby('user_id_hash')
	.agg({'coupon_id_hash': 'unique'})
	.reset_index())
interactions_train_dict = pd.Series(df_interactions_train.coupon_id_hash.values,
	index=df_interactions_train.user_id_hash).to_dict()

# let's load the activity matrix and dict of indexes
interactions_mtx = load_npz(os.path.join(inp_dir, train_dir, "interactions_mtx.npz"))

# We built the matrix as user x items, but for knn item based CF we need items x users
interactions_mtx_knn = interactions_mtx.T
# Let's build the KNN model...two lines :)
model = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
model.fit(interactions_mtx_knn)

# users and items indexes
items_idx_dict = pickle.load(open(os.path.join(inp_dir, train_dir, "items_idx_dict.p"),'rb'))
users_idx_dict = pickle.load(open(os.path.join(inp_dir, train_dir, "users_idx_dict.p"),'rb'))
idx_item_dict = {v:k for k,v in items_idx_dict.items()}

# validation activities
df_purchases_valid = pd.read_pickle(os.path.join(inp_dir, 'valid', 'df_purchases_valid.p'))
df_visits_valid = pd.read_pickle(os.path.join(inp_dir, 'valid', 'df_visits_valid.p'))
df_visits_valid.rename(index=str, columns={'view_coupon_id_hash': 'coupon_id_hash'}, inplace=True)

# interactions in validation: here we will not treat differently purchases or
# viewed. If we recommend and it was viewed or purchased, we will considered
# it as a hit
id_cols = ['user_id_hash', 'coupon_id_hash']
df_interactions_valid = pd.concat([df_purchases_valid[id_cols], df_visits_valid[id_cols]])
df_interactions_valid = df_interactions_valid[df_interactions_valid.user_id_hash.isin(users_idx_dict.keys())]
df_interactions_valid = (df_interactions_valid.groupby('user_id_hash')
	.agg({'coupon_id_hash': 'unique'})
	.reset_index())
tmp_valid_dict = pd.Series(df_interactions_valid.coupon_id_hash.values,
	index=df_interactions_valid.user_id_hash).to_dict()

# As before, we will concentrate in users that interacted at least with 1
# validation coupon during validation.
valid_coupon_ids = df_coupons_valid_feat.coupon_id_hash.values
keep_users = []
for user, coupons in tmp_valid_dict.items():
	if np.intersect1d(valid_coupon_ids, coupons).size !=0:
		keep_users.append(user)
interactions_valid_dict = {k:v for k,v in tmp_valid_dict.items() if k in keep_users}
del(tmp_valid_dict)

user_items_tuple = [(k,v) for k,v in interactions_valid_dict.items()]

def build_recommendations(user):
	coupons = interactions_train_dict[user]
	idxs = [items_idx_dict[c] for c in coupons]
	dist, nnidx = model.kneighbors(interactions_mtx_knn[idxs], n_neighbors = 11)
	dist, nnidx = dist[:, 1:], nnidx[:,1:]
	dist, nnidx = dist.ravel(), nnidx.ravel()
	ranked_dist = np.argsort(dist)
	ranked_cp_idxs = nnidx[ranked_dist][:50]
	ranked_cp_ids  = [idx_item_dict[i] for i in ranked_cp_idxs]
	ranked_cp_idxs_valid = [train_to_valid_most_similar[c] for c in ranked_cp_ids]
	return (user,ranked_cp_idxs_valid)

start = time()

cores = multiprocessing.cpu_count()

pool = Pool(cores)
all_users = list(interactions_valid_dict.keys())
recommend_coupons = pool.map(build_recommendations, all_users)

# recommend_coupons = Parallel(cores)(delayed(build_recommendations)(user) for user,_ in user_items_tuple)

print(time()-start)

recommendations_dict = {k:v for k,v in recommend_coupons}
actual = []
pred = []
for k,_ in recommendations_dict.items():
	actual.append(list(interactions_valid_dict[k]))
	pred.append(list(recommendations_dict[k]))

result = mapk(actual, pred)
print(result)