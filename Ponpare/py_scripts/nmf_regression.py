import numpy as np
import pandas as pd
import os
import pickle
import multiprocessing
import lightgbm as lgb

from joblib import Parallel, delayed
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.decomposition import NMF
from scipy.sparse import csr_matrix, load_npz
from sklearn.neighbors import NearestNeighbors
from recutils.average_precision import mapk
from sklearn.model_selection import train_test_split

inp_dir = "../datasets/Ponpare/data_processed/"
train_dir = "train"
valid_dir = "valid"

# train and validation coupons
df_coupons_train_feat = pd.read_pickle(os.path.join(inp_dir, train_dir, 'df_coupons_train_feat.p'))
df_coupons_valid_feat = pd.read_pickle(os.path.join(inp_dir, valid_dir, 'df_coupons_valid_feat.p'))
coupons_train_ids = df_coupons_train_feat.coupon_id_hash.values
coupons_valid_ids = df_coupons_valid_feat.coupon_id_hash.values

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

nmf_model = NMF(n_components=50, init='random', random_state=1981)
user_factors = nmf_model.fit_transform(interactions_mtx)
item_factors = nmf_model.components_.T

# make sure every user/item points to the right factors
user_factors_dict = {}
for k,v in users_idx_dict.items():
	user_factors_dict[k] = user_factors[users_idx_dict[k]]

item_factors_dict = {}
for k,v in items_idx_dict.items():
	item_factors_dict[k] = item_factors[items_idx_dict[k]]

df_interest = pd.read_pickle(os.path.join(inp_dir, train_dir, 'df_interest.p'))
df_user_factors = (pd.DataFrame.from_dict(user_factors_dict, orient="index")
	.reset_index())
df_user_factors.columns = ['user_id_hash'] + ['user_factor_'+str(i) for i in range(50)]
df_item_factors = (pd.DataFrame.from_dict(item_factors_dict, orient="index")
	.reset_index())
df_item_factors.columns = ['coupon_id_hash'] + ['item_factor_'+str(i) for i in range(50)]
df_train = pd.merge(df_interest[['user_id_hash','coupon_id_hash','interest']],
	df_item_factors, on='coupon_id_hash')
df_train = pd.merge(df_train, df_user_factors, on='user_id_hash')
X = df_train.iloc[:,3:].values
y = df_train.interest.values

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25)
model = lgb.LGBMRegressor(n_estimators=1000)
model.fit(X_train,y_train,
	eval_set = [(X_valid,y_valid)],
	early_stopping_rounds=20,
	eval_metric="rmse")

# VALIDATION
df_purchases_valid = pd.read_pickle(os.path.join(inp_dir, 'valid', 'df_purchases_valid.p'))
df_visits_valid = pd.read_pickle(os.path.join(inp_dir, 'valid', 'df_visits_valid.p'))
df_visits_valid.rename(index=str, columns={'view_coupon_id_hash': 'coupon_id_hash'}, inplace=True)

# subset users that were seeing in training
train_users = df_interest.user_id_hash.unique()
df_vva = df_visits_valid[df_visits_valid.user_id_hash.isin(train_users)]
df_pva = df_purchases_valid[df_purchases_valid.user_id_hash.isin(train_users)]

# interactions in validation: here we will not treat differently purchases or
# viewed. If we recommend and it was viewed or purchased, we will considered
# it as a hit
id_cols = ['user_id_hash', 'coupon_id_hash']
df_interactions_valid = pd.concat([df_pva[id_cols], df_vva[id_cols]], ignore_index=True)
df_interactions_valid = (df_interactions_valid.groupby('user_id_hash')
    .agg({'coupon_id_hash': 'unique'})
    .reset_index())
tmp_valid_dict = pd.Series(df_interactions_valid.coupon_id_hash.values,
    index=df_interactions_valid.user_id_hash).to_dict()

valid_coupon_ids = df_coupons_valid_feat.coupon_id_hash.values
keep_users = []
for user, coupons in tmp_valid_dict.items():
    if np.intersect1d(valid_coupon_ids, coupons).size !=0:
        keep_users.append(user)
# out of 6923, we end up with 6070, so not bad
interactions_valid_dict = {k:v for k,v in tmp_valid_dict.items() if k in keep_users}

# Take the 358 validation coupons and the 6070 users seen in training and during
# validation and rank!
left = pd.DataFrame({'user_id_hash':list(interactions_valid_dict.keys())})
left['key'] = 0
right = df_coupons_valid_feat[['coupon_id_hash']]
right['key'] = 0
df_valid = (pd.merge(left, right, on='key', how='outer')
    .drop('key', axis=1))

df_valid['mapped_coupons'] = (df_valid.coupon_id_hash
	.apply(lambda x: valid_to_train_most_similar[x]))
df_valid = pd.merge(df_valid, df_item_factors,
	left_on='mapped_coupons', right_on='coupon_id_hash')
df_valid = pd.merge(df_valid, df_user_factors,
	on='user_id_hash')

X_valid = df_valid.iloc[:, 4:].values
preds = model.predict(X_valid)

df_valid['interest'] = preds
df_preds = df_valid[['user_id_hash', 'coupon_id_hash_x', 'interest']]
df_preds.columns = ['user_id_hash', 'coupon_id_hash', 'interest']

df_ranked = df_preds.sort_values(['user_id_hash', 'interest'], ascending=[False, False])
df_ranked = (df_ranked
	.groupby('user_id_hash')['coupon_id_hash']
	.apply(list)
	.reset_index())
recomendations_dict = pd.Series(df_ranked.coupon_id_hash.values,
	index=df_ranked.user_id_hash).to_dict()

actual = []
pred = []
for k,_ in recomendations_dict.items():
	actual.append(list(interactions_valid_dict[k]))
	pred.append(list(recomendations_dict[k]))

print(mapk(actual,pred))
