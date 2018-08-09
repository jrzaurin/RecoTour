import pandas as pd
import numpy as np
import os

from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import MinMaxScaler
from recutils.average_precision import mapk

inp_dir = "../datasets/Ponpare/data_processed/"
train_dir = "train"
valid_dir = "valid"

# training interactions
df_purchases_train = pd.read_pickle(os.path.join(inp_dir, 'train', 'df_purchases_train.p'))
df_visits_train = pd.read_pickle(os.path.join(inp_dir, 'train', 'df_visits_train.p'))
df_visits_train.rename(index=str, columns={'view_coupon_id_hash': 'coupon_id_hash'}, inplace=True)

# train users
df_user_train_feat = pd.read_pickle(os.path.join(inp_dir, 'train', 'df_user_train_feat.p'))
train_users = df_user_train_feat.user_id_hash.unique()

# Compute popularity:
# popularity = n_purchaes + 0.1*n_visits
df_n_purchases = (df_purchases_train
	.coupon_id_hash
	.value_counts()
	.reset_index())
df_n_purchases.columns = ['coupon_id_hash','counts']
df_n_visits = (df_visits_train
	.coupon_id_hash
	.value_counts()
	.reset_index())
df_n_visits.columns = ['coupon_id_hash','counts']

# We will prioritise purchased coupons
df_popularity = df_n_purchases.merge(df_n_visits, on='coupon_id_hash', how='left')
df_popularity.fillna(0, inplace=True)
df_popularity['popularity'] = df_popularity['counts_x'] + 0.1*df_popularity['counts_y']
df_popularity.sort_values('popularity', ascending=False , inplace=True)

# Because none of the validation coupons have been seen during training we
# need to find a "proxy" for popularity. Here again one has freedom. Here we
# will compute validation popularity as the 1 minus mean distance between a
# validation coupon and the top10 most popular coupons during training
top10 = df_popularity.coupon_id_hash.tolist()[:10]

# top10 coupons features
df_coupons_train_feat = pd.read_pickle(os.path.join(inp_dir, 'train', 'df_coupons_train_feat.p'))
df_top_10_feat = (df_coupons_train_feat[df_coupons_train_feat.coupon_id_hash.isin(top10)]
	.reset_index())

# let's compute the popularity of validation coupons by calculating their
# distance to the most popular coupons during training
df_coupons_valid_feat = pd.read_pickle(os.path.join(inp_dir, 'valid', 'df_coupons_valid_feat.p'))
coupons_valid_ids = df_coupons_valid_feat.coupon_id_hash.values
cat_cols = [c for c in df_coupons_train_feat.columns if '_cat' in c]
id_cols = ['coupon_id_hash']
num_cols = [c for c in df_coupons_train_feat.columns if
	(c not in cat_cols) and (c not in id_cols)]

df_top_10_feat['flag'] = 0
df_coupons_valid_feat['flag'] = 1
tmp_df = pd.concat([
	df_top_10_feat[cat_cols+['flag']],
	df_coupons_valid_feat[cat_cols+['flag']]
	],
	ignore_index=True)
df_dummy_feats = pd.get_dummies(tmp_df, columns=cat_cols)

df_top_10_feat_oh = (df_dummy_feats[df_dummy_feats.flag == 0]
	.drop('flag', axis=1)
	.values)
coupons_valid_feat_oh = (df_dummy_feats[df_dummy_feats.flag == 1]
	.drop('flag', axis=1)
	.values)
del(tmp_df, df_dummy_feats)

df_top_10_feat_num = df_top_10_feat[num_cols].values
coupons_valid_feat_num = df_coupons_valid_feat[num_cols].values

scaler = MinMaxScaler()
df_top_10_feat_num_norm = scaler.fit_transform(df_top_10_feat_num)
coupons_valid_feat_num_norm = scaler.transform(coupons_valid_feat_num)

df_top_10_feat = np.hstack([df_top_10_feat_num_norm, df_top_10_feat_oh])
coupons_valid_feat = np.hstack([coupons_valid_feat_num_norm, coupons_valid_feat_oh])

euc_dist = pairwise_distances(coupons_valid_feat_num_norm, df_top_10_feat_num_norm, metric='euclidean')
jacc_dist = pairwise_distances(coupons_valid_feat_oh, df_top_10_feat_oh, metric='jaccard')

euc_dist_interp = np.empty((euc_dist.shape[0],euc_dist.shape[1]))
for i,(e,j) in enumerate(zip(euc_dist, jacc_dist)):
	l1,r1,l2,r2 = np.min(e), np.max(e), np.min(j), np.max(j)
	euc_dist_interp[i,:] = np.interp(e, [l1,r1], [l2,r2])
dist_mtx = (jacc_dist + euc_dist_interp)/2.

mean_distances = np.apply_along_axis(np.mean, 1, dist_mtx)
df_valid_popularity = pd.DataFrame({'coupon_id_hash': coupons_valid_ids,
	'popularity': 1-mean_distances})

# validation activities
df_purchases_valid = pd.read_pickle(os.path.join(inp_dir, 'valid', 'df_purchases_valid.p'))
df_visits_valid = pd.read_pickle(os.path.join(inp_dir, 'valid', 'df_visits_valid.p'))
df_visits_valid.rename(index=str, columns={'view_coupon_id_hash': 'coupon_id_hash'}, inplace=True)

# subset users that were seeing in training
df_vva = df_visits_valid[df_visits_valid.user_id_hash.isin(train_users)]
df_pva = df_purchases_valid[df_purchases_valid.user_id_hash.isin(train_users)]

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
# out of 6924, we end up with 6071, so not bad
interactions_valid_dict = {k:v for k,v in tmp_valid_dict.items() if k in keep_users}


# Take the 358 validation coupons and the 7057 users in total
left = pd.DataFrame({'user_id_hash':list(interactions_valid_dict.keys())})
left['key'] = 0
right = df_coupons_valid_feat[['coupon_id_hash']]
right['key'] = 0
df_valid = (pd.merge(left, right, on='key', how='outer')
	.drop('key', axis=1))
df_valid = pd.merge(df_valid, df_valid_popularity, on='coupon_id_hash')

df_ranked = df_valid.sort_values(['user_id_hash', 'popularity'], ascending=[False, False])
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


