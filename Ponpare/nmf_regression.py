import numpy as np
import pandas as pd
import os
import pickle
import multiprocessing

from joblib import Parallel, delayed
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.decomposition import NMF
from scipy.sparse import csr_matrix, load_npz
from sklearn.neighbors import NearestNeighbors
from recutils.average_precision import mapk


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

# Categorical: let's add a flag for convenience
df_coupons_train_feat['flag_cat'] = 0
df_coupons_valid_feat['flag_cat'] = 1

cat_cols = [c for c in df_coupons_train_feat.columns if c.endswith('_cat')]
id_cols = ['coupon_id_hash']
num_cols = [c for c in df_coupons_train_feat.columns if
	(c not in cat_cols) and (c not in id_cols)]

flag_cols = ['flag_cat_0','flag_cat_1']

tmp_df = pd.concat([df_coupons_train_feat[cat_cols],
	df_coupons_valid_feat[cat_cols]],
	ignore_index=True)
df_dummy_feats = pd.get_dummies(tmp_df.astype('category'))
del(tmp_df)

coupons_train_feat_oh = (df_dummy_feats[df_dummy_feats.flag_cat_0 != 0]
	.drop(flag_cols, axis=1)
	.values)
coupons_valid_feat_oh = (df_dummy_feats[df_dummy_feats.flag_cat_1 != 0]
	.drop(flag_cols, axis=1)
	.values)

# Numerical
df_coupons_train_feat['flag_num'] = 0
df_coupons_valid_feat['flag_num'] = 1

tmp_df = pd.concat([ df_coupons_train_feat[num_cols+['flag_num']],
	df_coupons_valid_feat[num_cols+['flag_num']] ],
	ignore_index=True)
df_num_feat_norm = (tmp_df[num_cols]-tmp_df[num_cols].min())/(tmp_df[num_cols].max()-tmp_df[num_cols].min())
df_num_feat_norm['flag_num'] = tmp_df['flag_num']
del(tmp_df)

coupons_train_feat_num = (df_num_feat_norm[df_num_feat_norm.flag_num == 0]
	.drop('flag_num', axis=1)
	.values)
coupons_valid_feat_num = (df_num_feat_norm[df_num_feat_norm.flag_num == 1]
	.drop('flag_num', axis=1)
	.values)

euc_dist = pairwise_distances(coupons_train_feat_num, coupons_valid_feat_num, metric='euclidean')
jacc_dist = pairwise_distances(coupons_train_feat_oh, coupons_valid_feat_oh, metric='jaccard')

euc_dist_interp = np.empty((euc_dist.shape[0],euc_dist.shape[1]))
for i,(e,j) in enumerate(zip(euc_dist, jacc_dist)):
	l1,r1,l2,r2 = np.min(e), np.max(e), np.min(j), np.max(j)
	euc_dist_interp[i,:] = np.interp(e, [l1,r1], [l2,r2])
tot_dist = (jacc_dist + euc_dist_interp)/2.

# now we have a matrix of distances, let's build the dictionaries
train_to_valid_top_n_idx = np.apply_along_axis(np.argsort, 1, tot_dist)
valid_to_train_top_n_idx = np.apply_along_axis(np.argsort, 1, tot_dist.T)
train_to_valid_most_similar = dict(zip(coupons_train_ids,
	coupons_valid_ids[train_to_valid_top_n_idx[:,0]]))
# there is one coupon in validation: '0a8e967835e2c20ac4ed8e69ee3d7349' that
# is never among the most similar to those previously seen.
valid_to_train_most_similar = dict(zip(coupons_valid_ids,
	coupons_train_ids[valid_to_train_top_n_idx[:,0]]))

# let's load the activity matrix and dict of indexes
interactions_mtx = load_npz(os.path.join(inp_dir, train_dir, "interactions_mtx.npz"))
items_idx_dict = pickle.load(open(os.path.join(inp_dir, train_dir, "items_idx_dict.p"),'rb'))
users_idx_dict = pickle.load(open(os.path.join(inp_dir, train_dir, "users_idx_dict.p"),'rb'))

nmf_model = NMF(n_components=10, init='random', random_state=1981)
user_factors = nmf_model.fit_transform(interactions_mtx)
item_factors = nmf_model.components_.T

# make sure every user/item points to the right factors
user_factors_dict = {}
for hash_id,idx in users_idx_dict.items():
	user_factors_dict[hash_id] = user_factors[users_idx_dict[hash_id]]

item_factors_dict = {}
for hash_id,idx in items_idx_dict.items():
	item_factors_dict[hash_id] = item_factors[items_idx_dict[hash_id]]
