import pandas as pd
import numpy as np
import os

from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import MinMaxScaler
from recutils.average_precision import mapk

inp_dir = "../datasets/Ponpare/data_processed/"
train_dir = "train"
valid_dir = "valid"

# train users
df_user_train_feat = pd.read_pickle(os.path.join(inp_dir, 'train', 'df_user_train_feat.p'))
train_users = df_user_train_feat.user_id_hash.unique()

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

coupon_id_rn = valid_coupon_ids.copy()
recomendations_dict = {}
for user, _  in interactions_valid_dict.items():
	np.random.shuffle(coupon_id_rn)
	recomendations_dict[user] = coupon_id_rn

actual = []
pred = []
for k,_ in recomendations_dict.items():
	actual.append(list(interactions_valid_dict[k]))
	pred.append(list(recomendations_dict[k]))

print(mapk(actual,pred))


