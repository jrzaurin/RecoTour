import numpy as np
import pandas as pd
import os
import xgboost as xgb
import lightgbm as lgb
import catboost as ctb
import warnings
import multiprocessing

from recutils.average_precision import mapk
from functools import reduce
from hyperopt import hp, tpe
from hyperopt.fmin import fmin

warnings.filterwarnings("ignore")
cores = multiprocessing.cpu_count()


def lgb_objective(params):
	"""
	objective function for lightgbm.
	"""

	lgb_objective.i+=1

	# hyperopt casts as float
	params['num_boost_round'] = int(params['num_boost_round'])
	params['num_leaves'] = int(params['num_leaves'])

	# need to be passed as parameter
	params['verbose'] = -1
	params['seed'] = 1

	cv_result = lgb.cv(
		params,
		lgtrain,
		nfold=3,
		metrics='rmse',
		num_boost_round=params['num_boost_round'],
		early_stopping_rounds=20,
		stratified=False,
		)

	error = cv_result['rmse-mean'][-1]
	print("INFO: iteration {} error {:.3f}".format(lgb_objective.i, error))

	return error


# This approach will be perhaps the one easier to understand. We have user
# features, item features and a target (interest), so let's turn this into a
# supervised problem and fit a regressor. Since this is a "standard" technique
# I will use this opportunity to illustrate a variety of tools around ML in
# general and boosted methods in particular

inp_dir = "../datasets/Ponpare/data_processed/"
train_dir = "train"
valid_dir = "valid"

# train coupon features
df_coupons_train_feat = pd.read_pickle(os.path.join(inp_dir, train_dir, 'df_coupons_train_feat.p'))

# In general, when using boosted methods, the presence of correlated or
# redundant features is not a big deal, since these will be ignored through
# the boosting rounds. However, for clarity and to reduce the chances of
# overfitting, we will select a subset of features here. All the numerical
# features have corresponding categorical ones, so we will keep those in
# moving forward. In addition, if we remember, for valid period, validend and
# validfrom, we used to methods to inpute NaN. Method1: Considering NaN as
# another category and Method2: replace NaN first in the object/numeric column
# and then turning the column into categorical. To start with, we will use
# Method1 here.
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

# for the time being we ignore recency
df_train.drop(['user_id_hash','coupon_id_hash','recency_factor'], axis=1, inplace=True)
train = df_train.drop('interest', axis=1)
y_train = df_train.interest.values
all_cols = train.columns.tolist()
cat_cols = [c for c in train.columns if '_cat' in c]

lgtrain = lgb.Dataset(train,
	label=y_train,
	feature_name=all_cols,
	categorical_feature = cat_cols,
	free_raw_data=False)

lgb_parameter_space = {
	'learning_rate': hp.uniform('learning_rate', 0.01, 0.5),
	'num_boost_round': hp.quniform('num_boost_round', 50, 500, 50),
	'num_leaves': hp.quniform('num_leaves', 30,1024,5),
    'min_child_weight': hp.quniform('min_child_weight', 1, 50, 2),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1.),
    'subsample': hp.uniform('subsample', 0.5, 1.),
    'reg_alpha': hp.uniform('reg_alpha', 0.01, 1.),
    'reg_lambda': hp.uniform('reg_lambda', 0.01, 1.),
}

lgb_objective.i = 0
best = fmin(fn=lgb_objective,
            space=lgb_parameter_space,
            algo=tpe.suggest,
            max_evals=10)
best['num_boost_round'] = int(best['num_boost_round'])
best['num_leaves'] = int(best['num_leaves'])
best['verbose'] = -1

inp_params = best.copy()
cv_result = lgb.cv(
	inp_params,
	lgtrain,
	nfold=3,
	metrics='rmse',
	num_boost_round=inp_params['num_boost_round'],
	early_stopping_rounds=20,
	stratified=False,
	)
best['num_boost_round'] = len(cv_result['rmse-mean'])

# validation activities
df_purchases_valid = pd.read_pickle(os.path.join(inp_dir, 'valid', 'df_purchases_valid.p'))
df_visits_valid = pd.read_pickle(os.path.join(inp_dir, 'valid', 'df_visits_valid.p'))
df_visits_valid.rename(index=str, columns={'view_coupon_id_hash': 'coupon_id_hash'}, inplace=True)

# Read the validation coupon features
df_coupons_valid_feat = pd.read_pickle(os.path.join(inp_dir, 'valid', 'df_coupons_valid_feat.p'))
df_coupons_valid_cat_feat = df_coupons_valid_feat.drop(drop_cols, axis=1)

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
df_valid = pd.merge(df_valid, df_users_train_feat, on='user_id_hash')
df_valid = pd.merge(df_valid, df_coupons_valid_cat_feat, on = 'coupon_id_hash')
X_valid = (df_valid
	.drop(['user_id_hash','coupon_id_hash'], axis=1)
	.values)

mod = lgb.train(best, lgtrain, feature_name=all_cols, categorical_feature=cat_cols)
preds = mod.predict(X_valid)

df_preds = df_valid[['user_id_hash','coupon_id_hash']]
df_preds['interest'] = preds

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

