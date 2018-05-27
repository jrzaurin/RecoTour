import numpy as np
import pandas as pd
import os
import xgboost as xgb
import lightgbm as lgb
import catboost as ctb
import warnings
import multiprocessing

from functools import reduce
from hyperopt import hp, tpe
from hyperopt.fmin import fmin
from sklearn.datasets import dump_svmlight_file

warnings.filterwarnings("ignore")
cores = multiprocessing.cpu_count()

inp_dir = "../datasets/Ponpare/data_processed/"
train_dir = "train"
valid_dir = "valid"

# train coupon features
df_coupons_train_feat = pd.read_pickle(os.path.join(inp_dir, train_dir, 'df_coupons_train_feat.p'))

drop_cols = [c for c in df_coupons_train_feat.columns
	if (('_cat' not in c) or ('method2' in c)) and (c!='coupon_id_hash')]
df_coupons_train_cat_feat = df_coupons_train_feat.drop(drop_cols, axis=1)

# train user features: there are a lot of features for users, both, numerical
# and categorical. We keep them all
df_users_train_feat = pd.read_pickle(os.path.join(inp_dir, train_dir, 'df_user_train_feat.p'))

# interest dataframe
df_interest = pd.read_pickle(os.path.join(inp_dir, train_dir, 'df_interest.p'))

df_train = pd.merge(df_interest, df_users_train_feat, on='user_id_hash')
df_train = pd.merge(df_train, df_coupons_train_cat_feat, on = 'coupon_id_hash')
df_obs_per_user = (df_train.groupby('user_id_hash')[['user_id_hash']]
	.size()
	.reset_index())
df_obs_per_user.columns = ['user_id_hash','n_obs']
df_train = df_train.merge(df_obs_per_user, on='user_id_hash')
# just to make sure
df_train_svm = df_train.sort_values('user_id_hash')

# There seems to be a bug in the cv method for lambdarank
# https://github.com/Microsoft/LightGBM/pull/397, so we are going to do this
# manually. We can' just sample observations at random, we need to sample
# users. Therefore, this will take some computational time, but at least we
# will be able to optimize parameters

# FROM HERE


# Observations per user (query group)
q_train = (df_train_svm[['user_id_hash','n_obs']]
	.drop_duplicates()['n_obs']).values.astype('int')

# For lambda_rank target needs to be categorical
interest_bins = np.percentile(df_train_svm.interest, q=[50, 75, 90])
df_train_svm['interest_rank'] = df_train_svm.interest.apply(
	lambda x: 0
	if x<=interest_bins[0]
	else 1 if ((x>interest_bins[0]) and (x<=interest_bins[1]))
	else 2
	)
df_train_svm.drop(['user_id_hash','coupon_id_hash','recency_factor','n_obs','interest'], axis=1, inplace=True)
all_cols = df_train_svm.columns.tolist()
cat_cols = [c for c in df_train_svm.columns if '_cat' in c]
X_train = df_train_svm.drop('interest_rank', axis=1).values
y_train = df_train_svm.interest_rank.values
all_cols = [c for c in all_cols if c!='interest_rank']




lgtrain = lgb.Dataset(
	train,
	label=y_train,
	group=q_train,
	feature_name=all_cols,
	categorical_feature=cat_cols,
	free_raw_data=False)

train = df_train_svm.drop('interest_rank', axis=1)
mod = lgb.LGBMRanker()
mod.fit(X_train, y_train, eval_set=[(X_train, y_train)], eval_group=[q_train], group=q_train, feature_name=all_cols, categorical_feature=cat_cols)
