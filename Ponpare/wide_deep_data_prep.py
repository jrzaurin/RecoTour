import numpy as np
import pandas as pd
import os
import pickle
import multiprocessing
import re

from joblib import Parallel, delayed
from recutils.average_precision import mapk
from recutils.utils import coupon_similarity_function, validation_interactions
from sklearn.model_selection import train_test_split
from collections import namedtuple

inp_dir = "../datasets/Ponpare/data_processed/"
train_dir = "train"
valid_dir = "valid"

train_coupons_path = os.path.join(inp_dir, train_dir, 'df_coupons_train_feat.p')
valid_coupons_path = os.path.join(inp_dir, valid_dir, 'df_coupons_valid_feat.p')

train_to_valid_most_similar, valid_to_train_most_similar = coupon_similarity_function(
	train_coupons_path,
	valid_coupons_path)

# train coupon features: approach_1: with coupon distance metric and user/item embeddings
df_coupons_train_feat = pd.read_pickle(os.path.join(train_coupons_path))
train_coupons = df_coupons_train_feat.coupon_id_hash.unique()
drop_cols = [c for c in df_coupons_train_feat.columns
	if (('_cat' not in c) or ('method2' in c)) and (c!='coupon_id_hash')]
df_coupons_feat = df_coupons_train_feat.drop(drop_cols, axis=1)

# user features
df_users_feat = pd.read_pickle(os.path.join(inp_dir, train_dir, 'df_user_train_feat.p'))

# let's add a categorical feature for age that will be used later for the crossed_colums
df_users_feat['age_cat'], age_bins = pd.qcut(df_users_feat['age'], q=4, labels=[0,1,2,3], retbins=True)

# coupons are all categorical columns and users have both.
coupons_cat_cols = [c for c in df_coupons_feat.columns if c != "coupon_id_hash"]
users_cat_cols = [c for c in df_users_feat.columns if c.endswith("_cat")]
users_num_cols = [c for c in df_users_feat.columns if
	(c not in users_cat_cols) and (c != "user_id_hash")]

tmp_df = df_users_feat[users_num_cols]
tmp_df_norm = (tmp_df-tmp_df.min())/(tmp_df.max()-tmp_df.min())
df_users_feat.drop(users_num_cols, axis=1, inplace=True)
df_users_feat = pd.concat([tmp_df_norm, df_users_feat], axis=1)
del(tmp_df, tmp_df_norm)

# let's decide whether to pass through the wide or deep side based on the
# levels of each categorical feature
coupon_cat_levels = {}
for c in coupons_cat_cols:
	n_levels = df_coupons_feat[c].nunique()
	coupon_cat_levels[c] = n_levels

user_cat_levels = {}
for c in users_cat_cols:
	n_levels = df_users_feat[c].nunique()
	user_cat_levels[c] = n_levels

# if a categorical feature has more than 10 levels we will use embeddings. The
# number of embeddings will depends on the number of levels
embeddings_cols=[]
for col,nl in coupon_cat_levels.items():
	if (nl >= 10) and (nl <=20): embeddings_cols.append((col, 6))
	elif (nl >= 10) and (nl > 20): embeddings_cols.append((col, 8))
for col,nl in user_cat_levels.items():
	if (nl >= 10) and (nl <=20): embeddings_cols.append((col, 6))
	elif (nl >= 10) and (nl > 20): embeddings_cols.append((col, 8))
embeddings_cols+=[('user_id_hash',10),('coupon_id_hash', 10)]

continuous_cols = users_num_cols.copy()

wide_cols = [c for c in coupons_cat_cols+users_cat_cols if c not in embeddings_cols]

# We can also pass crossed_colums (explain what they are and why is a bit of
# an incovenient to mix coupons and users feat)
cross_cols = (['sex_id_cat','top1_genre_name_cat'], ['age_cat', 'top1_genre_name_cat'])

# now we have all the info, let's build the datasets
if type(embeddings_cols[0]) is tuple:
    emb_dim = dict(embeddings_cols)
    embeddings_cols = [emb[0] for emb in embeddings_cols]
else:
    emb_dim = {e:def_dim for e in embeddings_cols}

# We will add embeddings for users and items
deep_cols = embeddings_cols+continuous_cols

def paste_cols(row, cols):
	vals = []
	for c in cols:
		vals.append(str(row[c]))
	return '-'.join(vals)

crossed_columns = []
for cols in cross_cols:
    colname = '_'.join(cols)
    df_users_feat[colname] = df_users_feat.apply(lambda x: paste_cols(x, cols), axis=1)
    crossed_columns.append(colname)
	wide_cols+=[colname]

# now we need to adapt a bit the dictionary of mappings we built "a while ago"
# so that we can recover embeddings
dict_of_mappings = pickle.load(open("../datasets/Ponpare/data_processed/dict_of_mappings.p", "rb"))

# We see that embeddings_cols mostly comprised locations and coupon category
# let's do the following so we can generalize later

# first, all embedding columns are categorical, so they will have "_cat" as a
# suffix, but different prefixes. Also, pref_name is the users dataset was
# mapped using ken_name, so will add an exception for that
encoding_dict = {}
prefixes = ['top1_', 'top2_', 'top3_']
for col in embeddings_cols:
	if ('pref_name' not in col) and ('id_hash' not in col):
		start=0
		for prefix in prefixes:
			if prefix in col:
				start = re.search(prefix, col).end()
		end = re.search("_cat", col).start()
		col_root_name = col[start:end]
		if col_root_name == "small_area": col_root_name = "small_area_name"
		encoding_dict[col] = dict_of_mappings[col_root_name]
# encoding_dict['pref_name_cat'] = dict_of_mappings['ken_name']

# Now it comes an important part, we are going to pass as embeddings users and items
dict_of_users = {k:v for v,k in enumerate(df_users_feat.user_id_hash.unique())}
dict_of_items = {k:v for v,k in enumerate(df_coupons_feat.coupon_id_hash.unique())}
encoding_dict['user_id_hash'] = dict_of_users
encoding_dict['coupon_id_hash'] = dict_of_items

embeddings_input = []
for k,v in encoding_dict.items():
    embeddings_input.append((k, len(v), emb_dim[k]))

# Now we need to build the training and validation datasets. Here we go:
df_interest = pd.read_pickle(os.path.join(inp_dir, train_dir, 'df_interest.p'))
df_interest.drop('recency_factor', axis=1, inplace=True)
df_train = pd.merge(df_interest, df_coupons_feat, on = 'coupon_id_hash')
df_train = pd.merge(df_train, df_users_feat, on='user_id_hash')

# Remember, we need to map user_id_hash and coupon_id_hash
df_train['user_id_hash'] = df_train.user_id_hash.apply(lambda x: dict_of_users[x])
df_train['coupon_id_hash'] = df_train.coupon_id_hash.apply(lambda x: dict_of_items[x])

# prepare parameters for validation_interactions function
purchases_path = os.path.join(inp_dir, 'valid', 'df_purchases_valid.p')
visist_path = os.path.join(inp_dir, 'valid', 'df_visits_valid.p')
train_users = df_interest.user_id_hash.unique()

interactions_valid_dict, df_valid = validation_interactions(
	purchases_path,
	visist_path,
	valid_coupons_path,
	train_users,
	drop_cols,
	valid_to_train_most_similar)

df_valid = pd.merge(df_valid, df_coupons_feat, on = 'coupon_id_hash')
df_valid = pd.merge(df_valid, df_users_feat, on='user_id_hash')

# again we need to map users and coupons
df_valid['user_id_hash'] = df_valid.user_id_hash.apply(lambda x: dict_of_users[x])
df_valid['coupon_id_hash'] = df_valid.coupon_id_hash.apply(lambda x: dict_of_items[x])

# now we need to split the datasets and save the indexes of the embedding columns
# split the training dataset into training and validation and use the
# validation as test dataset
df_train_tr, df_train_val = train_test_split(df_train, test_size=0.3, random_state=1981)
y_train, y_valid = df_train_tr.interest.values, df_train_val.interest.values

# extract deep columns
df_train_deep, df_valid_deep, df_test_deep = df_train_tr[deep_cols], df_train_val[deep_cols], df_valid[deep_cols]
deep_column_idx = {k:v for v,k in enumerate(df_train_deep.columns)}

# one hot encoding for the wide columns must be done all at once to ensure
# same number of dimensions through all datasets
df_train_tr['set_type']  = 0
df_train_val['set_type']  = 1
df_valid['set_type']  = 2
tmpdf = pd.concat([
	df_train_tr[wide_cols+['set_type']],
	df_train_val[wide_cols+['set_type']],
	df_valid[wide_cols+['set_type']]
	],
	ignore_index=True)
tmpdf_dummy = pd.get_dummies(tmpdf, columns=wide_cols)

df_train_wide = (tmpdf_dummy[tmpdf_dummy.set_type == 0]
	.drop('set_type', axis=1))
df_valid_wide = (tmpdf_dummy[tmpdf_dummy.set_type == 1]
	.drop('set_type', axis=1))
df_test_wide = (tmpdf_dummy[tmpdf_dummy.set_type == 2]
	.drop('set_type', axis=1))
del(tmpdf_dummy)

X_train_wide, X_train_deep = df_train_wide.values, df_train_deep.values
X_valid_wide, X_valid_deep = df_valid_wide.values, df_valid_deep.values
X_test_wide, X_test_deep   = df_test_wide.values, df_test_deep.values

# And finally lets create a convenient dictionary with all that we need
wd_dataset = dict()
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
wd_dataset['continuous_cols'] = continuous_cols

# pickle.dump(wd_dataset, open("wd_dataset.p", "wb"))