'''
There is more code in the corresponding notebook. This trully is an experimentation script
'''
import numpy as np
import pandas as pd
import pickle
import random
import os
# import xgboost as xgb
import lightgbm as lgb
import catboost as ctb
import warnings
import multiprocessing

from recutils.average_precision import mapk
from functools import reduce
from hyperopt import hp, tpe, fmin, Trials
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor

warnings.filterwarnings("ignore")


def lgb_objective(params):
	"""
	objective function for lightgbm.
	"""

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

	early_stop_dict[lgb_objective.i] = len(cv_result['rmse-mean'])
	error = cv_result['rmse-mean'][-1]
	print("INFO: iteration {} error {:.3f}".format(lgb_objective.i, error))

	lgb_objective.i+=1

	return error


def lgb_objective_map(params):
	"""
	objective function for lightgbm.
	"""

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
	early_stop_dict[lgb_objective_map.i] = len(cv_result['rmse-mean'])
	params['num_boost_round'] = len(cv_result['rmse-mean'])

	model = lgb.LGBMRegressor(**params)
	model.fit(train,y_train,feature_name=all_cols,categorical_feature=cat_cols)
	preds = model.predict(X_valid)

	df_eval['interest'] = preds
	df_ranked = df_eval.sort_values(['user_id_hash', 'interest'], ascending=[False, False])
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

	result = mapk(actual,pred)
	print("INFO: iteration {} MAP {:.3f}".format(lgb_objective_map.i, result))

	lgb_objective_map.i+=1

	return 1-result


inp_dir = "/home/ubuntu/projects/RecoTour/datasets/Ponpare/data_processed/"
train_dir = "train"
valid_dir = "valid"

# train coupon features
df_coupons_train_feat = pd.read_pickle(os.path.join(inp_dir, train_dir, 'df_coupons_train_feat.p'))

drop_cols = [c for c in df_coupons_train_feat.columns
    if ((not c.endswith('_cat')) or ('method2' in c)) and (c!='coupon_id_hash')]
df_coupons_train_cat_feat = df_coupons_train_feat.drop(drop_cols, axis=1)

# train user features
df_users_train_feat = pd.read_pickle(os.path.join(inp_dir, train_dir, 'df_users_train_feat.p'))

# interest dataframe
df_interest = pd.read_pickle(os.path.join(inp_dir, train_dir, 'df_interest.p'))

df_train = pd.merge(df_interest, df_users_train_feat, on='user_id_hash')
df_train = pd.merge(df_train, df_coupons_train_cat_feat, on = 'coupon_id_hash')

# df_train['interest'] = df_train['interest'] * df_train['recency_factor']

# for the time being we ignore recency
df_train.drop(['user_id_hash','coupon_id_hash','recency_factor'], axis=1, inplace=True)
train = df_train.drop('interest', axis=1)
y_train = df_train.interest
all_cols = train.columns.tolist()
cat_cols = [c for c in train.columns if c.endswith("_cat")]

# lgb dataset object
lgtrain = lgb.Dataset(train,
	label=y_train,
	feature_name=all_cols,
	categorical_feature = cat_cols,
	free_raw_data=False)

# Read the validation coupon features
df_coupons_valid_feat = pd.read_pickle(os.path.join(inp_dir, 'valid', 'df_coupons_valid_feat.p'))
df_coupons_valid_cat_feat = df_coupons_valid_feat.drop(drop_cols, axis=1)

# Read the interactions during validation
interactions_valid_dict = pickle.load(open(inp_dir+"/valid/interactions_valid_dict.p", "rb"))

# Build a validation dataframe with the cartesian product between the 358 validation coupons
# and the 6071 users seen in training AND validation
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
df_eval = df_valid[['user_id_hash','coupon_id_hash']]

# defining the parameter space
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

early_stop_dict = {}
trials = Trials()
lgb_objective_map.i = 0
best = fmin(fn=lgb_objective_map,
            space=lgb_parameter_space,
            algo=tpe.suggest,
            max_evals=10,
            trials=trials)
best['num_boost_round'] = early_stop_dict[trials.best_trial['tid']]
best['num_leaves'] = int(best['num_leaves'])
best['verbose'] = -1
print(1-trials.best_trial['result']['loss'])

model = lgb.LGBMRegressor(**best)
model.fit(train,y_train,feature_name=all_cols,categorical_feature=cat_cols)
# save model
# model.booster_.save_model('model.txt')
# to load
# model = lgb.Booster(model_file='mode.txt')

preds = model.predict(X_valid)
df_eval['interest'] = preds
df_ranked = df_eval.sort_values(['user_id_hash', 'interest'], ascending=[False, False])
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

# -----------------------------------------------------------------------------
# MODEL INTERPRETABILITY (see corresponding notebook)

# -----------------------------------------------------------------------------
# EXPERIMENTING WITH SKLEARN'S LIGHTGBM EQUIVALENT
def hgb_objective_map(params):
	"""
	objective function for HistGradientBoostingRegressor.
	"""

	# hyperopt casts as float
	params['max_iter'] = int(params['max_iter'])
	params['max_leaf_nodes'] = int(params['max_leaf_nodes'])

	model = HistGradientBoostingRegressor(**params)
	model.fit(train,y_train)
	preds = model.predict(X_valid)

	df_eval['interest'] = preds
	df_ranked = df_eval.sort_values(['user_id_hash', 'interest'], ascending=[False, False])
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

	result = mapk(actual,pred)
	print("INFO: iteration {} MAP {:.3f}".format(lgb_objective_map.i, result))

	hgb_objective_map.i+=1

	return 1-result

# defining the parameter space
hgb_parameter_space = {
	'learning_rate': hp.uniform('learning_rate', 0.01, 0.5),
	'max_iter': hp.quniform('max_iter', 50, 500, 50),
	'max_leaf_nodes': hp.quniform('max_leaf_nodes', 30,1024,5),
    'l2_regularization': hp.uniform('l2_regularization', 0.01, 1.)
}

hgb_objective_map.i = 0
best = fmin(fn=hgb_objective_map,
            space=hgb_parameter_space,
            algo=tpe.suggest,
            max_evals=10)
