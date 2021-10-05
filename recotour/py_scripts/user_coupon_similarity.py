import pandas as pd
import numpy as np
import os

from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
from sklearn.metrics import jaccard_similarity_score
from recutils.average_precision import mapk
from hyperopt import hp, tpe
from hyperopt.fmin import fmin
from skopt import gbrt_minimize

# Here we will represent each user by the average of coupons they purchased or
# visit. There is a lot of freedom on how we weight the different coupons in
# the mean, or how one could combine purchases with visit. For example, one
# can weight the coupons based on the amount of items they bought with them.
# One can consider only unique coupons in the purchase and visit table or we
# can consider coupons that were used more than once as some form of weight.

# For the excercise here we will consider unique purchased and viewed coupons
# and we will combine them using a weight parameter that we will optimize
# against the Mean Average Precision

inp_dir = "/home/ubuntu/projects/RecoTour/datasets/Ponpare/data_processed/"
train_dir = "train"
valid_dir = "valid"

# training interactions
df_purchases_train = pd.read_pickle(os.path.join(inp_dir, 'train', 'df_purchases_train.p'))
df_visits_train = pd.read_pickle(os.path.join(inp_dir, 'train', 'df_visits_train.p'))
df_visits_train.rename(index=str, columns={'view_coupon_id_hash': 'coupon_id_hash'}, inplace=True)

# train users and coupons features
df_coupons_train_feat = pd.read_pickle(os.path.join(inp_dir, 'train', 'df_coupons_train_feat.p'))
df_user_train_feat = pd.read_pickle(os.path.join(inp_dir, 'train', 'df_users_train_feat.p'))
train_users = df_user_train_feat.user_id_hash.unique()
train_coupons = df_coupons_train_feat.coupon_id_hash.unique()

# subset activities according to the users seen in training
df_vtr = df_visits_train[df_visits_train.user_id_hash.isin(train_users) &
	df_visits_train.coupon_id_hash.isin(train_coupons)]
df_ptr = df_purchases_train[df_purchases_train.user_id_hash.isin(train_users) &
	df_purchases_train.coupon_id_hash.isin(train_coupons)]

# Most recent purchase is the purchase that will be considered for the mean
df_ptr_most_recent = (df_ptr
	.groupby(['user_id_hash','coupon_id_hash'])['days_to_present']
	.min()
	.reset_index())
df_ptr_most_recent.drop('days_to_present', axis=1, inplace=True)

# In order to build features we will separare coupons that were purchased and
# viewed. This means that if a coupon was viewed and then purchased, only the
# purchase interaction will be considered when computed the mean. Also, as
# mentioned before, if someone "interacted with" that coupon many times, here
# we will only consider it once
df_vtr_visits = df_vtr.copy()
df_vtr_visits['activity_hash'] = df_vtr_visits['user_id_hash'] + "_" + df_vtr_visits['coupon_id_hash']
purchases = df_vtr_visits[~df_vtr_visits.purchaseid_hash.isna()]['activity_hash'].unique()
df_vtr_visits = (df_vtr_visits[~df_vtr_visits.activity_hash
	.isin(purchases)][['user_id_hash','coupon_id_hash','days_to_present']])

# Most recent visit is the view that will be considered
df_vtr_most_recent = (df_vtr_visits
	.groupby(['user_id_hash','coupon_id_hash'])['days_to_present']
	.min()
	.reset_index())
df_vtr_most_recent.drop('days_to_present', axis=1, inplace=True)

# Merge with coupon features
df_ptr_most_recent = (df_ptr_most_recent
	.merge(df_coupons_train_feat, on='coupon_id_hash',how='left'))

# # Two possibilites: directly aggregate features through the mean, or one-hot
# # encode 1st. The former is easier and leads to almost identical results. Also
# # requires for all the columns to be converted to integer for the mean to be
# # calculated. To do the former simply uncomment the lines below and comment
# # the one-hot encoding block
# cols_to_int = [c for c in df_ptr_most_recent.columns if 'id_hash' not in c]
# df_ptr_most_recent[cols_to_int] = df_ptr_most_recent[cols_to_int].astype('int')
# df_vtr_most_recent = (df_vtr_most_recent
# 	.merge(df_coupons_train_feat, on='coupon_id_hash',how='left'))
# df_vtr_most_recent[cols_to_int] = df_vtr_most_recent[cols_to_int].astype('int')
df_vtr_most_recent = (df_vtr_most_recent
	.merge(df_coupons_train_feat, on='coupon_id_hash',how='left'))


# Calculate the mean feature vectors for purchases and views. one-hot encode
# categorical columns. To get dummies all cols have to be categorical. ALSO,
# and very importantly, when compute the one hot encoded features, we need to
# do it "all at once", training and validation together. This is to ensure
# that when calculating distances, all vectors will have the same number of
# elements

# Read the validation coupons
df_coupons_valid_feat = pd.read_pickle(os.path.join(inp_dir, 'valid', 'df_coupons_valid_feat.p'))

# for convenience, we will flag each dataset
df_ptr_most_recent['flag_cat'] = 0
df_vtr_most_recent['flag_cat'] = 1
df_coupons_valid_feat['flag_cat'] = 2
flag_cols = ['flag_cat_0','flag_cat_1','flag_cat_2']

cat_cols = [c for c in df_ptr_most_recent.columns if '_cat' in c]
non_cat_cols = [c for c in df_ptr_most_recent.columns if c not in cat_cols]
# no user_id in df_coupons_valid_feat
non_cat_cols_valid = [c for c in non_cat_cols if c != 'user_id_hash']

tmp_df = pd.concat([df_ptr_most_recent[cat_cols],
	df_vtr_most_recent[cat_cols],
	df_coupons_valid_feat[cat_cols]],
	ignore_index=True)

df_dummy_feats = pd.get_dummies(tmp_df.astype('category'))

tmp_df_ptr_oh = (df_dummy_feats[df_dummy_feats.flag_cat_0 != 0]
	.drop(flag_cols, axis=1)
	.reset_index(drop=True))
tmp_df_vtr_oh = (df_dummy_feats[df_dummy_feats.flag_cat_1 != 0]
	.drop(flag_cols, axis=1)
	.reset_index(drop=True))
tmp_df_valid_oh = (df_dummy_feats[df_dummy_feats.flag_cat_2 != 0]
	.drop(flag_cols, axis=1)
	.reset_index(drop=True))

df_ptr_most_recent_oh = pd.concat([df_ptr_most_recent[non_cat_cols],
	tmp_df_ptr_oh], axis=1)
df_vtr_most_recent_oh = pd.concat([df_vtr_most_recent[non_cat_cols],
	tmp_df_vtr_oh], axis=1)
df_coupons_valid_feat_oh = pd.concat([df_coupons_valid_feat[non_cat_cols_valid],
	tmp_df_valid_oh], axis=1)

# mean vectors
user_mean_purchase_vector = (df_ptr_most_recent_oh.groupby('user_id_hash')
	.mean()
	.reset_index())
user_mean_visit_vector = (df_vtr_most_recent_oh.groupby('user_id_hash')
	.mean()
	.reset_index())

# validation activities
df_purchases_valid = pd.read_pickle(os.path.join(inp_dir, 'valid', 'df_purchases_valid.p'))
df_visits_valid = pd.read_pickle(os.path.join(inp_dir, 'valid', 'df_visits_valid.p'))
df_visits_valid.rename(index=str, columns={'view_coupon_id_hash': 'coupon_id_hash'}, inplace=True)

# subset users that were seeing in training
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

# We have 358 coupons that were displayed during validation. However, users
# seen at training interacted with 2078 coupons. These are coupons that were
# displayed before but can still be used during validation. As a consequence,
# there might me users that, during validation they have no recorded
# interaction with validation coupons. If we include every single interaction,
# then we will have a slighly negative view on how well we are performing.
# Therefore, for the time being, we will concentrate in users that were seen
# during training AND interacted at least with 1 validation coupon during
# validation.

# There are a number of additional caveats one should consider if this was
# "real life". For example, for those coupons that were not used, is it
# because they were seen and not liked? or not seen? When a customer views a
# coupon but never uses it, is it because they did not like it? and in
# consequence we might want to include negative feedback?

# For now, let's do the simple things (if we do not one-hot encode, we can
# just read df_coupons_valid_feat here)
valid_coupon_ids = df_coupons_valid_feat.coupon_id_hash.values

keep_users = []
for user, coupons in tmp_valid_dict.items():
	if np.intersect1d(valid_coupon_ids, coupons).size !=0:
		keep_users.append(user)
# out of 6924, we end up with 6071, so not bad
interactions_valid_dict = {k:v for k,v in tmp_valid_dict.items() if k in keep_users}

# 6071 users that were seen in training that are also seen during validation
# 358 coupons that were displayed during validation and we need to recommend
# Therefore, we need to compute the similarity between those 6071 users and
# 358 coupons, and recommend the top N most similar.

# Subset the vectors of features for the users that were seen in validation
user_mean_purchase_vector_valid = (user_mean_purchase_vector[user_mean_purchase_vector
	.user_id_hash
	.isin(interactions_valid_dict.keys())]
	.reset_index(drop=True))
user_mean_visit_vector_valid = (user_mean_visit_vector[user_mean_visit_vector
	.user_id_hash
	.isin(interactions_valid_dict.keys())]
	.reset_index(drop=True))
users_valid = pd.concat([user_mean_purchase_vector_valid,
	user_mean_visit_vector_valid])['user_id_hash'].unique()
# we lose one user that visited one coupon and that coupon not in
# train_coupons. So 6070 users and 358 coupons
lost_user = [usr for usr in interactions_valid_dict.keys() if usr not in users_valid]
del interactions_valid_dict[lost_user[0]]

def mapk_similarity(alpha, at_random=False):

	mpv = user_mean_purchase_vector_valid.copy()
	feat_cols = [c for c in mpv.columns if 'id_hash' not in c]
	mvv = user_mean_visit_vector_valid.copy()
	mvv[feat_cols] = alpha*mvv[feat_cols]

	user_vector= (pd.concat([mpv, mvv])
		.groupby('user_id_hash')
		.sum()
		.reset_index())

	user_ids = user_vector.user_id_hash.values
	item_ids = df_coupons_valid_feat_oh.coupon_id_hash.values
	# ensure the same column order
	user_cols = ['user_id_hash'] + [c for c in user_vector.columns if 'id_hash' not in c]
	item_cols = ['coupon_id_hash'] + [c for c in user_vector.columns if 'id_hash' not in c]
	user_feat = user_vector[user_cols[1:]].values
	item_feat = df_coupons_valid_feat_oh[item_cols[1:]].values

	user_item_sim = euclidean_distances(user_feat, item_feat)
	top_n_idx = np.apply_along_axis(np.argsort, 1, user_item_sim)

	if at_random:
		item_feat_rnd = item_ids.copy()
		recomendations_dict = {}
		for user,idx in zip(user_ids,top_n_idx):
			np.random.shuffle(item_feat_rnd)
			recomendations_dict[user] = item_feat_rnd
	else:
		recomendations_dict = {}
		for user,idx in zip(user_ids,top_n_idx):
			recomendations_dict[user] = [item_ids[i] for i in idx]

	actual = []
	pred = []
	for k,_ in recomendations_dict.items():
		actual.append(list(interactions_valid_dict[k]))
		pred.append(list(recomendations_dict[k]))

	return mapk(actual, pred)


def sim_objective(params, method="hyperopt"):

	if method is "hyperopt":
		sim_objective.i+=1
		alpha = params['alpha']
	elif method is "skopt":
		alpha = params[0]

	score = mapk_similarity(alpha)

	if method is "hyperopt":
		print("INFO: iteration {} error {:.3f}".format(sim_objective.i, score))

	return 1-score

partial_objective = lambda params: sim_objective(params,
	method=method)

hp_params = {'alpha': hp.uniform('alpha', 0.01, 1.)}
method = "hyperopt"
sim_objective.i=0
hp_best = fmin(fn=partial_objective,
	space=hp_params,
	algo=tpe.suggest,
	max_evals=100)
print(hp_best)
print(mapk_similarity(hp_best['alpha']))

sk_params = [(0.01, 1, 'uniform')]
method = "skopt"
sk_best = gbrt_minimize(partial_objective,
	sk_params,
	n_calls=100,
	random_state=0,
	verbose=False,
	n_jobs=-1)
print(sk_best.x)
print(1-sk_best.fun)
