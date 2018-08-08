import pandas as pd
import numpy as np
import pickle
import os

from collections import Counter
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import MinMaxScaler


def coupon_similarity_function(train_coupons_path, valid_coupons_path, method="cosine"):

	df_coupons_train_feat = pd.read_pickle(train_coupons_path)
	df_coupons_valid_feat = pd.read_pickle(valid_coupons_path)

	coupons_train_ids = df_coupons_train_feat.coupon_id_hash.values
	coupons_valid_ids = df_coupons_valid_feat.coupon_id_hash.values

	df_coupons_train_feat['flag'] = 0
	df_coupons_valid_feat['flag'] = 1

	cat_cols = [c for c in df_coupons_train_feat.columns if c.endswith('_cat')]
	id_cols = ['coupon_id_hash']
	num_cols = [c for c in df_coupons_train_feat.columns if
		(c not in cat_cols) and (c not in id_cols)]

	tmp_df = pd.concat([
		df_coupons_train_feat[cat_cols+['flag']],
		df_coupons_valid_feat[cat_cols+['flag']]
		],
		ignore_index=True)

	df_dummy_feats = pd.get_dummies(tmp_df, columns=cat_cols)

	coupons_train_feat_oh = (df_dummy_feats[df_dummy_feats.flag == 0]
		.drop('flag', axis=1)
		.values)
	coupons_valid_feat_oh = (df_dummy_feats[df_dummy_feats.flag == 1]
		.drop('flag', axis=1)
		.values)
	del(tmp_df, df_dummy_feats)

	coupons_train_feat_num = df_coupons_train_feat[num_cols].values
	coupons_valid_feat_num = df_coupons_valid_feat[num_cols].values

	scaler = MinMaxScaler()
	coupons_train_feat_num_norm = scaler.fit_transform(coupons_train_feat_num)
	coupons_valid_feat_num_norm = scaler.transform(coupons_valid_feat_num)

	coupons_train_feat = np.hstack([coupons_train_feat_num_norm, coupons_train_feat_oh])
	coupons_valid_feat = np.hstack([coupons_valid_feat_num_norm, coupons_valid_feat_oh])

	if method is "cosine":
		dist_mtx = pairwise_distances(coupons_valid_feat, coupons_train_feat, metric='cosine')
		valid_to_train_top_n_idx = np.apply_along_axis(np.argsort, 1, dist_mtx)
		valid_to_train_most_similar = dict(zip(coupons_valid_ids,
			coupons_train_ids[valid_to_train_top_n_idx[:,0]]))

	elif method is "combined":

		euc_dist = pairwise_distances(coupons_valid_feat_num_norm, coupons_train_feat_num_norm, metric='euclidean')
		jacc_dist = pairwise_distances(coupons_valid_feat_oh, coupons_train_feat_oh, metric='jaccard')

		euc_dist_interp = np.empty((euc_dist.shape[0],euc_dist.shape[1]))
		for i,(e,j) in enumerate(zip(euc_dist, jacc_dist)):
			l1,r1,l2,r2 = np.min(e), np.max(e), np.min(j), np.max(j)
			euc_dist_interp[i,:] = np.interp(e, [l1,r1], [l2,r2])
		dist_mtx = (jacc_dist + euc_dist_interp)/2.

		valid_to_train_top_n_idx = np.apply_along_axis(np.argsort, 1, dist_mtx)
		valid_to_train_most_similar = dict(zip(coupons_valid_ids,
			coupons_train_ids[valid_to_train_top_n_idx[:,0]]))

	return valid_to_train_most_similar


def validation_interactions(purchases_path, visist_path, valid_coupons_path, train_users, drop_cols,  mapping_dict=None):

	df_purchases_valid = pd.read_pickle(purchases_path)
	df_visits_valid = pd.read_pickle(visist_path)
	df_visits_valid.rename(index=str, columns={'view_coupon_id_hash': 'coupon_id_hash'}, inplace=True)

	df_coupons_valid_feat = pd.read_pickle(valid_coupons_path)
	df_coupons_valid_cat_feat = df_coupons_valid_feat.drop(drop_cols, axis=1)

	df_vva = df_visits_valid[df_visits_valid.user_id_hash.isin(train_users)]
	df_pva = df_purchases_valid[df_purchases_valid.user_id_hash.isin(train_users)]

	id_cols = ['user_id_hash', 'coupon_id_hash']
	df_interactions_valid = pd.concat([df_pva[id_cols], df_vva[id_cols]], ignore_index=True)
	df_interactions_valid = (df_interactions_valid.groupby('user_id_hash')
		.agg({'coupon_id_hash': 'unique'})
		.reset_index())
	tmp_valid_dict = pd.Series(df_interactions_valid.coupon_id_hash.values, index=df_interactions_valid.user_id_hash).to_dict()

	valid_coupon_ids = df_coupons_valid_feat.coupon_id_hash.values
	keep_users = []
	for user, coupons in tmp_valid_dict.items():
		if np.intersect1d(valid_coupon_ids, coupons).size !=0:
			keep_users.append(user)
	interactions_valid_dict = {k:v for k,v in tmp_valid_dict.items() if k in keep_users}

	if mapping_dict:
		df_coupons_valid_cat_feat['valid_coupon_id_hash'] = df_coupons_valid_cat_feat['coupon_id_hash']
		df_coupons_valid_cat_feat['coupon_id_hash'] = \
			df_coupons_valid_cat_feat.coupon_id_hash.apply(lambda x: mapping_dict[x])
		right = df_coupons_valid_cat_feat[['coupon_id_hash','valid_coupon_id_hash']]
	else:
		right = df_coupons_valid_cat_feat[['coupon_id_hash']]

	left = pd.DataFrame({'user_id_hash':list(interactions_valid_dict.keys())})
	left['key'] = 0
	right['key'] = 0
	df_valid = (pd.merge(left, right, on='key', how='outer')
		.drop('key', axis=1))

	return interactions_valid_dict, df_valid