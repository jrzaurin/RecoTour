import numpy as np
import pandas as pd
import os
import pickle
import lightgbm as lgb
import warnings
import multiprocessing
import random

from time import time
from hyperopt import hp, tpe
from hyperopt.fmin import fmin
from sklearn.model_selection import KFold
from recutils.average_precision import mapk

warnings.filterwarnings("ignore")
cores = multiprocessing.cpu_count()

inp_dir = "../datasets/Ponpare/data_processed/"
train_dir = "train"
valid_dir = "valid"

# train coupon features
df_coupons_train_feat = pd.read_pickle(os.path.join(inp_dir, train_dir, 'df_coupons_train_feat.p'))

# using just certain categorical features for coupon
drop_cols = [c for c in df_coupons_train_feat.columns
	if (('_cat' not in c) or ('method2' in c)) and (c!='coupon_id_hash')]
df_coupons_train_cat_feat = df_coupons_train_feat.drop(drop_cols, axis=1)

# train user features: there are a lot of features for users, both, numerical
# and categorical. We keep them all
df_users_train_feat = pd.read_pickle(os.path.join(inp_dir, train_dir, 'df_user_train_feat.p'))

# interest dataframe
df_interest = pd.read_pickle(os.path.join(inp_dir, train_dir, 'df_interest.p'))

# merge all together and count number of observations per user
df_train = pd.merge(df_interest, df_users_train_feat, on='user_id_hash')
df_train = pd.merge(df_train, df_coupons_train_cat_feat, on = 'coupon_id_hash')
df_obs_per_user = (df_train.groupby('user_id_hash')[['user_id_hash']]
	.size()
	.reset_index())
df_obs_per_user.columns = ['user_id_hash','n_obs']

# For lambda_rank target needs to be categorical
df_train['interest'] = df_train['interest'] * df_train['recency_factor']

interest_bins = np.percentile(df_train.interest, q=[50, 90, 95])
df_train['interest_rank'] = df_train.interest.apply(
	lambda x: 0
	if x<=interest_bins[0]
	else 1 if ((x>interest_bins[0]) and (x<=interest_bins[1]))
	else 2
	)
df_train.drop(
	['recency_factor','coupon_id_hash','interest'],
	axis=1, inplace=True)

all_cols = df_train.columns.tolist()
ignore_cols = ['user_id_hash','interest_rank']
all_cols = [c for c in all_cols if c not in ignore_cols]
cat_cols = [c for c in df_train.columns if c.endswith("_cat")]

# sort them to ensure same order
df_obs_per_user.sort_values('user_id_hash', inplace=True)
df_train.sort_values('user_id_hash', inplace=True)

def build_dataset(df_obs, df_feat, indexes):
	df_set_obs = df_obs.iloc[indexes, :]
	df_set_feat = df_feat[df_feat.user_id_hash.isin(df_set_obs.user_id_hash)]
	q_set = df_set_obs.n_obs.values
	y_set = df_set_feat.interest_rank.values
	final_set = (df_set_feat[all_cols], y_set, q_set)
	return final_set

def lgb_objective(params):

	lgb_objective.i+=1

	start = time()
	seed = random.randint(1,1000)
	kf = KFold(n_splits=3, shuffle=True, random_state=seed)
	train_sets, eval_sets = [],[]
	for train_index, eval_index in kf.split(df_obs_per_user):
		train_sets.append(build_dataset(df_obs_per_user, df_train, train_index))
		eval_sets.append(build_dataset(df_obs_per_user, df_train, eval_index))

	params['num_boost_round'] = int(params['num_boost_round'])
	params['num_leaves'] = int(params['num_leaves'])
	params['verbose'] = -1

	params['objective'] = 'lambdarank'
	params['metric'] = 'map'
	params['eval_at'] = 10

	scores=[]
	for train_set, eval_set in  zip(train_sets, train_sets):
		inp_params = params.copy()
		lgbtrain = lgb.Dataset(data=train_set[0],
			label=train_set[1],
			group=train_set[2],
			feature_name=all_cols,
			categorical_feature=cat_cols)
		lgbeval = lgb.Dataset(data=eval_set[0],
			label=eval_set[1],
			group=eval_set[2],
			reference=lgbtrain)
		mod = lgb.train(
			inp_params,
			lgbtrain,
			valid_sets=[lgbeval],
			early_stopping_rounds=10,
			verbose_eval=False)
		scores.append(mod.best_score['valid_0']['map@10'])

	score = np.mean(scores)
	end = time() - start
	print("INFO: iteration {} completed in {}. Score {:.3f}. ".format(lgb_objective.i, round(end,2), score))

	return 1-score

lgb_parameter_space = {
	'learning_rate': hp.uniform('learning_rate', 0.01, 0.5),
	'num_boost_round': hp.quniform('num_boost_round', 20, 100, 5),
	'num_leaves': hp.quniform('num_leaves', 32,256,4),
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

# Read the validation coupon features
df_coupons_valid_feat = pd.read_pickle(os.path.join(inp_dir, 'valid', 'df_coupons_valid_feat.p'))
df_coupons_valid_cat_feat = df_coupons_valid_feat.drop(drop_cols, axis=1)

# Read the interactions during validation
interactions_valid_dict = pickle.load(
    open("../datasets/Ponpare/data_processed/valid/interactions_valid_dict.p", "rb"))

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

lgbtrain = lgb.Dataset(data=df_train[all_cols],
	label=df_train.interest_rank,
	group=df_obs_per_user.n_obs,
	feature_name=all_cols,
	categorical_feature=cat_cols)

best['objective'] = 'lambdarank'
best['metric'] = 'map'

mod = lgb.train(best, lgbtrain, feature_name=all_cols, categorical_feature=cat_cols)
preds = mod.predict(df_valid[all_cols])

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

print(mapk(actual,pred))

