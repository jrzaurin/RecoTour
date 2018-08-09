import numpy as np
import pandas as pd
import pickle
import os
import lightgbm as lgb
import warnings
import multiprocessing

from joblib import Parallel, delayed
from recutils.average_precision import mapk
from functools import reduce
from hyperopt import hp, tpe, fmin, Trials

warnings.filterwarnings("ignore")
cores = multiprocessing.cpu_count()

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
		actual.append(list(interactions_test_dict[k]))
		pred.append(list(recomendations_dict[k]))

	result = mapk(actual,pred)
	print("INFO: iteration {} MAP {:.3f}".format(lgb_objective_map.i, result))

	lgb_objective_map.i+=1

	return 1-result

inp_dir = "../datasets/Ponpare/data_processed/"
train_dir = "ftrain"

# TRAIN DATASET
# train coupon features
df_coupons_train_feat = pd.read_pickle(os.path.join(inp_dir, train_dir, 'df_coupons_train_feat.p'))
drop_cols = [c for c in df_coupons_train_feat.columns
    if ((not c.endswith('_cat')) or ('method2' in c)) and (c!='coupon_id_hash')]
df_coupons_train_cat_feat = df_coupons_train_feat.drop(drop_cols, axis=1)

# train user features
df_users_train_feat = pd.read_pickle(os.path.join(inp_dir, train_dir, 'df_user_train_feat.p'))

# interest dataframe
df_interest = pd.read_pickle(os.path.join(inp_dir, train_dir, 'df_interest.p'))
train_users = df_interest.user_id_hash.unique()

df_train = pd.merge(df_interest, df_users_train_feat, on='user_id_hash')
df_train = pd.merge(df_train, df_coupons_train_cat_feat, on = 'coupon_id_hash')

# for the time being we ignore recency
df_train.drop(['user_id_hash','coupon_id_hash','recency_factor'], axis=1, inplace=True)
train = df_train.drop('interest', axis=1)
y_train = df_train.interest
all_cols = train.columns.tolist()
cat_cols = [c for c in train.columns if c.endswith("_cat")]

# Read the test coupon features
df_coupons_test_feat = pd.read_pickle(os.path.join(inp_dir, 'test', 'df_coupons_test_feat.p'))
df_coupons_test_cat_feat = df_coupons_test_feat.drop(drop_cols, axis=1)

# TEST DATASET
# test activities
df_purchases_test = pd.read_pickle(os.path.join(inp_dir, 'test', 'df_purchases_test.p'))
df_visits_test = pd.read_pickle(os.path.join(inp_dir, 'test', 'df_visits_test.p'))
df_visits_test.rename(index=str, columns={'view_coupon_id_hash': 'coupon_id_hash'}, inplace=True)

# subset users that were seeing in training
df_vte = df_visits_test[df_visits_test.user_id_hash.isin(train_users)]
df_pte = df_purchases_test[df_purchases_test.user_id_hash.isin(train_users)]

# dictionary of interactions to evaluate
id_cols = ['user_id_hash', 'coupon_id_hash']
df_interactions_test = pd.concat([df_pte[id_cols], df_vte[id_cols]], ignore_index=True)
df_interactions_test = (df_interactions_test.groupby('user_id_hash')
    .agg({'coupon_id_hash': 'unique'})
    .reset_index())
interactions_test_dict = pd.Series(df_interactions_test.coupon_id_hash.values,
    index=df_interactions_test.user_id_hash).to_dict()

# Build a dataframe with the cartesian product between the 433 test coupons
# and the 7295 users seen in training AND validation
left = pd.DataFrame({'user_id_hash':list(interactions_test_dict.keys())})
left['key'] = 0
right = df_coupons_test_feat[['coupon_id_hash']]
right['key'] = 0
df_valid = (pd.merge(left, right, on='key', how='outer')
    .drop('key', axis=1))
df_valid = pd.merge(df_valid, df_users_train_feat, on='user_id_hash')
df_valid = pd.merge(df_valid, df_coupons_test_cat_feat, on = 'coupon_id_hash')
X_valid = (df_valid
    .drop(['user_id_hash','coupon_id_hash'], axis=1)
    .values)
df_eval = df_valid[['user_id_hash','coupon_id_hash']]

# RUN THE EXPERIMENT

# lgb dataset object
lgtrain = lgb.Dataset(train,
	label=y_train,
	feature_name=all_cols,
	categorical_feature = cat_cols,
	free_raw_data=False)

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

# optimize against MAP
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


# fit model using best params
model = lgb.LGBMRegressor(**best)
%time model.fit(train,y_train,feature_name=all_cols,categorical_feature=cat_cols)
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
	actual.append(list(interactions_test_dict[k]))
	pred.append(list(recomendations_dict[k]))

print(mapk(actual,pred))