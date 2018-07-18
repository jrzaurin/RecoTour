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

import multiprocessing
from joblib import Parallel, delayed

# Here we will use KNN to build a item-based collaborative filtering
# recommendation algorithm. However, as straightforward this might sound,
# there is an issue to address in this and any purely interaction-based
# approach. We will build a matrix of interactions based on past interactions,
# but the new coupons have never been seen before. Therefore, in real life, as
# they come to the dataset and need to be displayed we could re-process and
# re-train, or we could use a distance metric and associate new items to the
# one existing. We will illustrate the later here.

inp_dir = "../datasets/Ponpare/data_processed/"
train_dir = "train"
valid_dir = "valid"
# train and validation coupons
df_coupons_train_feat = pd.read_pickle(os.path.join(inp_dir, train_dir, 'df_coupons_train_feat.p'))
df_coupons_valid_feat = pd.read_pickle(os.path.join(inp_dir, valid_dir, 'df_coupons_valid_feat.p'))
coupons_train_ids = df_coupons_train_feat.coupon_id_hash.values
coupons_valid_ids = df_coupons_valid_feat.coupon_id_hash.values


# let's build a dictionary of most similar coupon from both train->valid and
# valid->train. Here again we have a set of possibilities. One would be
# directly compare features, which of course brings the caveat that
# numerically comparing categorical features makes no sense, but since we just
# want THE CLOSESTS pairs of coupons, I think one could got away with this one
# here. Another one would be one-hot encode categorical features and then
# compute cosine distance for the whole set of features, categorical and
# numerical. And a final one, and the one we will explore here, will be
# compute the jaccard_similarity for the one-hot encoded features and
# euclidean "similarity" for the numerical one and combine them. (A further
# one would just be use only the categorical, since most of the "content" of
# the numerical features is (or should be) captured by the categorical

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
valid_to_train_top_n_idx = np.apply_along_axis(np.argsort, 1, dist_mtx)
valid_to_train_most_similar = dict(zip(coupons_valid_ids,
	coupons_train_ids[valid_to_train_top_n_idx[:,0]]))


# euc_dist = pairwise_distances(coupons_train_feat_num, coupons_valid_feat_num, metric='euclidean')
# jacc_dist = pairwise_distances(coupons_train_feat_oh, coupons_valid_feat_oh, metric='jaccard')

# euc_dist_interp = np.empty((euc_dist.shape[0],euc_dist.shape[1]))
# for i,(e,j) in enumerate(zip(euc_dist, jacc_dist)):
# 	l1,r1,l2,r2 = np.min(e), np.max(e), np.min(j), np.max(j)
# 	euc_dist_interp[i,:] = np.interp(e, [l1,r1], [l2,r2])
# tot_dist = (jacc_dist + euc_dist_interp)/2.

# now we have a matrix of distances, let's build the dictionaries
valid_to_train_top_n_idx = np.apply_along_axis(np.argsort, 1, dist_mtx)
train_to_valid_top_n_idx = np.apply_along_axis(np.argsort, 1, dist_mtx.T)
train_to_valid_most_similar = dict(zip(coupons_train_ids,
	coupons_valid_ids[train_to_valid_top_n_idx[:,0]]))
# there is one coupon in validation: '0a8e967835e2c20ac4ed8e69ee3d7349' that
# is never among the most similar to those previously seen.
valid_to_train_most_similar = dict(zip(coupons_valid_ids,
	coupons_train_ids[valid_to_train_top_n_idx[:,0]]))

# let's load the activity matrix and dict of indexes
interactions_mtx = load_npz(os.path.join(inp_dir, train_dir, "interactions_mtx.npz"))
# We built the matrix as user x items, but for knn item based CF we need items x users
interactions_mtx_knn = interactions_mtx.T

items_idx_dict = pickle.load(open(os.path.join(inp_dir, train_dir, "items_idx_dict.p"),'rb'))
idx_item_dict = {v:k for k,v in items_idx_dict.items()}

# The idea is:
# 1-. Per user during validation collect the interactions with coupons
# 2-. Per coupon, get the most similar training coupon and the
# corresponding N KNN based in the interactions_mtx.
# 3-. Rank them based on distance
# 4-. Map them back to validation coupons and recommend

# Note that here we based our recommendations on items. Therefore, in
# principle we can recommend to any user in validation
# Let's load validation activities
df_purchases_valid = pd.read_pickle(os.path.join(inp_dir, 'valid', 'df_purchases_valid.p'))
df_visits_valid = pd.read_pickle(os.path.join(inp_dir, 'valid', 'df_visits_valid.p'))
df_visits_valid.rename(index=str, columns={'view_coupon_id_hash': 'coupon_id_hash'}, inplace=True)

# interactions in validation: here we will not treat differently purchases or
# viewed. If we recommend and it was viewed or purchased, we will considered
# it as a hit
id_cols = ['user_id_hash', 'coupon_id_hash']
df_interactions_valid = pd.concat([df_purchases_valid[id_cols], df_visits_valid[id_cols]])
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

# Let's build the KNN model...two lines :)
model = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
model.fit(interactions_mtx_knn)

# Map the training interactions into validation. We run this loop on its own
# for convenience
interactions_mapped = {}
for user, coupons in interactions_valid_dict.items():
	# per user...
	mapped_coupons = []
	for coupon in coupons:
		# if the coupon is not among the training coupons...
		if coupon not in coupons_train_ids:
			try:
				# then map it to a training coupon (exception in case there
				# were no features for that coupon)
				coupon = valid_to_train_most_similar[coupon]
			except KeyError:
				continue
		mapped_coupons.append(coupon)
	interactions_mapped[user] = mapped_coupons

# let's put it in a tuple, and build a function to run it in Parallel
user_item_tuple = [(k,v) for k,v in interactions_mapped.items()]

def build_recommendations(user,coupons):
	# when ranking the most similar ones will be themselves -> ignore them
	ignore = len(coupons)
	# indexes in the matrix of interactions
	idxs = [items_idx_dict[c] for c in coupons]
	dist, nnidx = model.kneighbors(interactions_mtx_knn[idxs], n_neighbors = 11)
	dist, nnidx = dist.ravel(), nnidx.ravel()
	# rank indexes based on distance
	ranked_dist = np.argsort(dist)[ignore:]
	ranked_cp_idxs = nnidx[ranked_dist]
	ranked_train_cp = [idx_item_dict[i] for i in ranked_cp_idxs]
	# map training into validation coupons
	ranked_valid_cp = [train_to_valid_most_similar[c] for c in ranked_train_cp]
	return (user, ranked_valid_cp)


start = time()
cores = multiprocessing.cpu_count()
recommend_coupons = Parallel(n_jobs=cores)(delayed(build_recommendations)(user,coupons) for user,coupons in user_item_tuple)
print(time()-start)

recommendations_dict = {k:v for k,v in recommend_coupons}
actual = []
pred = []
for k,_ in recommendations_dict.items():
	actual.append(list(interactions_valid_dict[k]))
	pred.append(list(recommendations_dict[k]))

result = mapk(actual, pred)
print(result)