import numpy as np
import pandas as pd
import os
import pickle
import warnings
import multiprocessing
import lightgbm as lgb

from time import time
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.decomposition import NMF
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from scipy.sparse import csr_matrix, load_npz
from recutils.average_precision import mapk
from hyperopt import hp, tpe, fmin, Trials

warnings.filterwarnings("ignore")
cores = multiprocessing.cpu_count()

inp_dir = "../datasets/Ponpare/data_processed/"
train_dir = "train"
valid_dir = "valid"

# train and validation coupons
df_coupons_train_feat = pd.read_pickle(os.path.join(inp_dir, train_dir, 'df_coupons_train_feat.p'))
df_coupons_valid_feat = pd.read_pickle(os.path.join(inp_dir, valid_dir, 'df_coupons_valid_feat.p'))
coupons_train_ids = df_coupons_train_feat.coupon_id_hash.values
coupons_valid_ids = df_coupons_valid_feat.coupon_id_hash.values

id_cols = ['coupon_id_hash']
cat_cols = [c for c in df_coupons_train_feat.columns if c.endswith('_cat')]
num_cols = [c for c in df_coupons_train_feat.columns if
	(c not in cat_cols) and (c not in id_cols)]

df_coupons_train_feat['flag'] = 0
df_coupons_valid_feat['flag'] = 1

tmp_df = pd.concat(
	[df_coupons_train_feat,df_coupons_valid_feat],
	ignore_index=True)

# Normalize numerical columns
tmp_df_num = tmp_df[num_cols]
tmp_df_norm = (tmp_df_num-tmp_df_num.min())/(tmp_df_num.max()-tmp_df_num.min())
tmp_df[num_cols] = tmp_df_norm

# one hot categorical
tmp_df[cat_cols] = tmp_df[cat_cols].astype('category')
tmp_df_dummy = pd.get_dummies(tmp_df, columns=cat_cols)

coupons_train_feat = tmp_df_dummy[tmp_df_dummy.flag==0]
coupons_valid_feat = tmp_df_dummy[tmp_df_dummy.flag==1]
coupons_train_feat = (coupons_train_feat
	.drop(['flag','coupon_id_hash'], axis=1)
	.values)
coupons_valid_feat = (coupons_valid_feat
	.drop(['flag','coupon_id_hash'], axis=1)
	.values)

dist_mtx = pairwise_distances(coupons_valid_feat, coupons_train_feat, metric='cosine')
valid_to_train_top_n_idx = np.apply_along_axis(np.argsort, 1, dist_mtx)
valid_to_train_most_similar = dict(zip(coupons_valid_ids,
	coupons_train_ids[valid_to_train_top_n_idx[:,0]]))

# let's load the activity matrix and dict of indexes
interactions_mtx = load_npz(os.path.join(inp_dir, train_dir, "interactions_mtx.npz"))
items_idx_dict = pickle.load(open(os.path.join(inp_dir, train_dir, "items_idx_dict.p"),'rb'))
users_idx_dict = pickle.load(open(os.path.join(inp_dir, train_dir, "users_idx_dict.p"),'rb'))

ncomp = 100
nmf_model = NMF(n_components=ncomp, init='random', random_state=1981)
user_factors = nmf_model.fit_transform(interactions_mtx)
item_factors = nmf_model.components_.T
joblib.dump(nmf_model, "../datasets/Ponpare/data_processed/models/nmf_model.p")

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
df_user_factors.columns = ['user_id_hash'] + ['user_factor_'+str(i) for i in range(ncomp)]
df_item_factors = (pd.DataFrame.from_dict(item_factors_dict, orient="index")
	.reset_index())
df_item_factors.columns = ['coupon_id_hash'] + ['item_factor_'+str(i) for i in range(ncomp)]

# TRAIN
df_train = pd.merge(df_interest[['user_id_hash','coupon_id_hash','interest']],
	df_item_factors, on='coupon_id_hash')
df_train = pd.merge(df_train, df_user_factors, on='user_id_hash')
X = df_train.iloc[:,3:].values
y = df_train.interest.values

# VALIDATION
interactions_valid_dict = pickle.load(
	open("../datasets/Ponpare/data_processed/valid/interactions_valid_dict.p","rb"))
# remember that one user that visited one coupon and that coupon is not in the training set of coupons.
# and in consequence not in the interactions matrix
interactions_valid_dict.pop("25e2b645bfcd0980b2a5d0a4833f237a")

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
df_preds = df_valid[['user_id_hash', 'coupon_id_hash_x']]
df_preds.columns = ['user_id_hash', 'coupon_id_hash']

X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size=0.25)
model = lgb.LGBMRegressor(n_estimators=1000)
model.fit(X_train,y_train,
	eval_set = [(X_eval,y_eval)],
	early_stopping_rounds=10,
	eval_metric="rmse")

preds = model.predict(X_valid)

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

# def lgb_objective_map(params):
# 	"""
# 	objective function for lightgbm.
# 	"""

# 	# hyperopt casts as float
# 	params['num_boost_round'] = int(params['num_boost_round'])
# 	params['num_leaves'] = int(params['num_leaves'])

# 	# need to be passed as parameter
# 	params['verbose'] = -1
# 	params['seed'] = 1

# 	cv_result = lgb.cv(
# 	params,
# 	lgtrain,
# 	nfold=3,
# 	metrics='rmse',
# 	num_boost_round=params['num_boost_round'],
# 	early_stopping_rounds=20,
# 	stratified=False,
# 	)
# 	early_stop_dict[lgb_objective_map.i] = len(cv_result['rmse-mean'])
# 	params['num_boost_round'] = len(cv_result['rmse-mean'])

# 	model = lgb.LGBMRegressor(**params)
# 	model.fit(X,y)
# 	preds = model.predict(X_valid)

# 	df_preds['interest'] = preds
# 	df_ranked = df_preds.sort_values(['user_id_hash', 'interest'], ascending=[False, False])
# 	df_ranked = (df_ranked
# 		.groupby('user_id_hash')['coupon_id_hash']
# 		.apply(list)
# 		.reset_index())
# 	recomendations_dict = pd.Series(df_ranked.coupon_id_hash.values,
# 		index=df_ranked.user_id_hash).to_dict()

# 	actual = []
# 	pred = []
# 	for k,_ in recomendations_dict.items():
# 		actual.append(list(interactions_valid_dict[k]))
# 		pred.append(list(recomendations_dict[k]))

# 	result = mapk(actual,pred)
# 	print("INFO: iteration {} MAP {:.3f}".format(lgb_objective_map.i, result))

# 	lgb_objective_map.i+=1

# 	return 1-result

# # lgb dataset object
# lgtrain = lgb.Dataset(X,
# 	label=y,
# 	free_raw_data=False)

# # defining the parameter space
# lgb_parameter_space = {
# 	'learning_rate': hp.uniform('learning_rate', 0.01, 0.5),
# 	'num_boost_round': hp.quniform('num_boost_round', 100, 500, 50),
# 	'num_leaves': hp.quniform('num_leaves', 30,1024,5),
#     'min_child_weight': hp.quniform('min_child_weight', 1, 50, 2),
#     'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1.),
#     'subsample': hp.uniform('subsample', 0.5, 1.),
#     'reg_alpha': hp.uniform('reg_alpha', 0.01, 1.),
#     'reg_lambda': hp.uniform('reg_lambda', 0.01, 1.),
# }

# early_stop_dict = {}
# trials = Trials()
# start = time()
# lgb_objective_map.i = 0
# best = fmin(fn=lgb_objective_map,
#             space=lgb_parameter_space,
#             algo=tpe.suggest,
#             max_evals=50,
#             trials=trials)
# best['num_boost_round'] = early_stop_dict[trials.best_trial['tid']]
# best['num_leaves'] = int(best['num_leaves'])
# best['verbose'] = -1
# print(1-trials.best_trial['result']['loss'])
# print(time()-start)
# print(best)
# pickle.dump(best,
# 	open("../datasets/Ponpare/data_processed/models/gbm_nmf_optimal_parameters.p", "wb"))