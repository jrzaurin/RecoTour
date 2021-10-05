import numpy as np
import pandas as pd
import argparse
import pickle
import os
import re

from recutils.average_precision import mapk
from recutils.utils import coupon_similarity_function, validation_interactions
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split

def wd_prepare_data(train_dir, valid_dir, out_dir):

	# Create dictionaries mapping the training <-> validation coupons
	train_coupons_path = os.path.join(train_dir, 'df_coupons_train_feat.p')

	# train coupon features:
	# approach_3: numerical and categorical thorugh a linear side and user/item through the deep
	df_coupons_train_feat = pd.read_pickle(os.path.join(train_coupons_path))
	drop_cols = [c for c in df_coupons_train_feat.columns
		if (('_cat' not in c) or ('method2' in c)) and (c!='coupon_id_hash')]
	df_coupons_feat = df_coupons_train_feat.drop(drop_cols, axis=1)

	# user features
	df_users_feat = pd.read_pickle(os.path.join(train_dir, 'df_user_train_feat.p'))

	# let's add a categorical feature for age that will be used later for the crossed_colums
	df_users_feat['age_cat'], age_bins = pd.qcut(df_users_feat['age'], q=4, labels=[0,1,2,3], retbins=True)

	# let's update the dict_of_mappings
	dict_of_mappings = pickle.load(open("../datasets/Ponpare/data_processed/dict_of_mappings.p", "rb"))
	dict_of_mappings['age_cat'] = age_bins

	# coupons are all categorical columns and users have both.
	coupons_cat_cols = [c for c in df_coupons_feat.columns if c != "coupon_id_hash"]
	users_cat_cols = [c for c in df_users_feat.columns if c.endswith("_cat")]
	users_num_cols = [c for c in df_users_feat.columns if
		(c not in users_cat_cols) and (c != "user_id_hash")]

	# Normalize numerical columns
	tmp_df = df_users_feat[users_num_cols]
	tmp_df_norm = (tmp_df-tmp_df.min())/(tmp_df.max()-tmp_df.min())
	df_users_feat.drop(users_num_cols, axis=1, inplace=True)
	df_users_feat = pd.concat([tmp_df_norm, df_users_feat], axis=1)
	del(tmp_df, tmp_df_norm)

	# The linear/wide side of the model will be comprised by the one-hot
	# encoded categorical and continuous features. The deep side only user and
	# item embeddings

	# DEEP
	encoding_dict = {}
	user_dict = {k:v for v,k in enumerate(df_users_feat.user_id_hash.unique())}
	coupon_dict = {k:v for v,k in enumerate(df_coupons_feat.coupon_id_hash.unique())}
	encoding_dict['user_id_hash'] = user_dict
	encoding_dict['coupon_id_hash'] = coupon_dict
	embeddings_cols=[('user_id_hash',50), ('coupon_id_hash',50)]
	embeddings_input = [(col, len(encoding_dict[col]), n_emb) for (col, n_emb) in embeddings_cols]

	deep_cols = [c[0] for c in embeddings_cols]

	# WIDE: for the sake of memory we will assume that the user's numerical
	# features are already good in representing the user and we will only
	# consider 2 user categorical features
	wide_cont_cols = users_num_cols.copy()
	wide_cat_cols = [c for c in coupons_cat_cols+users_cat_cols[:2] if c not in deep_cols]

	# TRAIN DATASET
	df_interest = pd.read_pickle(os.path.join(train_dir, 'df_interest.p'))
	df_interest.drop('recency_factor', axis=1, inplace=True)
	df_train = pd.merge(df_interest, df_coupons_feat, on = 'coupon_id_hash')
	df_train = pd.merge(df_train, df_users_feat, on='user_id_hash')

	# VALIDATION DATASET
	# prepare parameters for validation_interactions function
	purchases_path = os.path.join(valid_dir, 'df_purchases_valid.p')
	valid_coupons_path = os.path.join(valid_dir, 'df_coupons_valid_feat.p')
	visist_path = os.path.join(valid_dir , 'df_visits_valid.p')
	train_users = df_interest.user_id_hash.unique()

	valid2train_most_similar = coupon_similarity_function(train_coupons_path,
		valid_coupons_path, method="combined")

	true_valid_interactions, df_valid = validation_interactions(
		purchases_path,
		visist_path,
		valid_coupons_path,
		train_users,
		drop_cols,
		valid2train_most_similar)

	df_valid = pd.merge(df_valid, df_coupons_feat, on='coupon_id_hash')
	df_valid = pd.merge(df_valid, df_users_feat, on='user_id_hash')

	# TRAIN/VALID/TEST split
	# First one hot encoding must be done all at once to ensure same number of
	# dimensions through all datasets
	df_train['set_type']  = 0
	df_valid['set_type']  = 1
	tmpdf = pd.concat([df_train,df_valid], ignore_index=True)
	tmpdf_dummy = pd.get_dummies(tmpdf, prefix="dummy", columns=wide_cat_cols)

	df_train_wd = (tmpdf_dummy[tmpdf_dummy.set_type == 0]
		.drop('set_type', axis=1))
	df_valid_wd = (tmpdf_dummy[tmpdf_dummy.set_type == 1]
		.drop('set_type', axis=1))
	del(tmpdf_dummy)

	hash_id_cols = ['user_id_hash', 'coupon_id_hash', 'valid_coupon_id_hash']

	dummy_cols_idx = [i for i,c in enumerate(df_train_wd.columns) if 'dummy' in c]
	cont_cols_idx = [i for i,c in enumerate(df_train_wd.columns) if c in wide_cont_cols]
	wide_cols_idx = cont_cols_idx+dummy_cols_idx

	df_tr, df_val = train_test_split(df_train_wd, test_size=0.3, random_state=1981)
	y_train, y_valid = df_tr.interest.values, df_val.interest.values

	# extract deep columns
	df_tr_deep, df_val_deep, df_test_deep = df_tr[deep_cols], df_val[deep_cols], df_valid_wd[deep_cols]
	deep_column_idx = {k:v for v,k in enumerate(df_tr_deep.columns)}
	df_tr_deep['user_id_hash'] = df_tr_deep.user_id_hash.apply(lambda x: user_dict[x])
	df_tr_deep['coupon_id_hash'] = df_tr_deep.coupon_id_hash.apply(lambda x: coupon_dict[x])
	df_val_deep['user_id_hash'] = df_val_deep.user_id_hash.apply(lambda x: user_dict[x])
	df_val_deep['coupon_id_hash'] = df_val_deep.coupon_id_hash.apply(lambda x: coupon_dict[x])
	df_test_deep['user_id_hash'] = df_test_deep.user_id_hash.apply(lambda x: user_dict[x])
	df_test_deep['coupon_id_hash'] = df_test_deep.coupon_id_hash.apply(lambda x: coupon_dict[x])

	df_tr_wide, df_val_wide, df_test_wide = (df_tr.iloc[:, wide_cols_idx],
		df_val.iloc[:, wide_cols_idx],
		df_valid_wd.iloc[:, wide_cols_idx])

	X_train_wide, X_train_deep = csr_matrix(df_tr_wide.values), df_tr_deep.values
	X_valid_wide, X_valid_deep = csr_matrix(df_val_wide.values), df_val_deep.values
	X_test_wide, X_test_deep   = csr_matrix(df_test_wide.values), df_test_deep.values

	# SAVE TO A DICT
	wd_dataset = {}
	wd_dataset['train_dataset'] = {}
	wd_dataset['train_dataset']['wide'], \
	wd_dataset['train_dataset']['deep'], \
	wd_dataset['train_dataset']['target'] = X_train_wide, X_train_deep, y_train
	wd_dataset['valid_dataset'] = {}
	wd_dataset['valid_dataset']['wide'], \
	wd_dataset['valid_dataset']['deep'], \
	wd_dataset['valid_dataset']['target'] = X_valid_wide, X_valid_deep, y_valid
	wd_dataset['test_dataset'] = {}
	wd_dataset['test_dataset']['wide'], \
	wd_dataset['test_dataset']['deep'] = X_test_wide, X_test_deep
	wd_dataset['embeddings_input']  = embeddings_input
	wd_dataset['deep_column_idx'] = deep_column_idx
	wd_dataset['encoding_dict'] = encoding_dict
	pickle.dump(wd_dataset, open(os.path.join(out_dir, "wd_dataset.p"), "wb"), protocol=4)

	interactions_dict = {}
	interactions_dict['true_valid_interactions'] = true_valid_interactions
	interactions_dict['all_valid_interactions'] = df_valid_wd[hash_id_cols]
	pickle.dump(interactions_dict, open(os.path.join(out_dir, "interactions_dict.p"), "wb"))

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description="build interaction datasets")

	parser.add_argument("--train_dir", type=str, default="../datasets/Ponpare/data_processed/train")
	parser.add_argument("--valid_dir", type=str, default="../datasets/Ponpare/data_processed/valid")
	parser.add_argument("--out_dir", type=str, default="../datasets/Ponpare/data_processed/wide_deep")

	args = parser.parse_args()

	wd_prepare_data(args.train_dir, args.valid_dir, args.out_dir)

