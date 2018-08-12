import numpy as np
import pandas as pd
import pickle
import os
import argparse
import lightgbm as lgb
import warnings
import multiprocessing

from recutils.utils import coupon_similarity_function
from recutils.average_precision import mapk
from sklearn.metrics.pairwise import pairwise_distances
from hyperopt import hp, tpe, fmin, Trials

warnings.filterwarnings("ignore")
cores = multiprocessing.cpu_count()


def top10_train_coupon_populatiry(train_purchases_path, train_visits_path):

	# train coupon popularity based on purchases and visits
	df_purchases_train = pd.read_pickle(train_purchases_path)
	df_visits_train = pd.read_pickle(train_visits_path)
	df_visits_train.rename(index=str, columns={'view_coupon_id_hash': 'coupon_id_hash'}, inplace=True)

	# popularity = n_purchases + 0.1*n_visits
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

	df_popularity = df_n_purchases.merge(df_n_visits, on='coupon_id_hash', how='left')
	df_popularity.fillna(0, inplace=True)
	df_popularity['popularity'] = df_popularity['counts_x'] + 0.1*df_popularity['counts_y']
	df_popularity.sort_values('popularity', ascending=False , inplace=True)

	# select top 10 most popular coupons from the training dataset
	top10 = df_popularity.coupon_id_hash.tolist()[:10]

	return top10


def test_coupon_populatiry(train_coupons_path, test_coupons_path):

	top10 = top10_train_coupon_populatiry(train_purchases_path, train_visits_path)

	# Compute distances between train and test coupons based on features
	df_coupons_train_feat = pd.read_pickle(train_coupons_path)
	df_coupons_test_feat = pd.read_pickle(test_coupons_path)
	coupons_test_ids = df_coupons_test_feat.coupon_id_hash.values

	id_cols = ['coupon_id_hash']
	cat_cols = [c for c in df_coupons_train_feat.columns if c.endswith('_cat')]
	num_cols = [c for c in df_coupons_train_feat.columns if
		(c not in cat_cols) and (c not in id_cols)]

	# Compute distances
	df_coupons_train_feat['flag'] = 0
	df_coupons_test_feat['flag'] = 1

	tmp_df = pd.concat(
		[df_coupons_train_feat,df_coupons_test_feat],
		ignore_index=True)

	# Normalize numerical columns
	tmp_df_num = tmp_df[num_cols]
	tmp_df_norm = (tmp_df_num-tmp_df_num.min())/(tmp_df_num.max()-tmp_df_num.min())
	tmp_df[num_cols] = tmp_df_norm

	# one hot categorical
	tmp_df[cat_cols] = tmp_df[cat_cols].astype('category')
	tmp_df_dummy = pd.get_dummies(tmp_df, columns=cat_cols)
	coupons_train_feat = tmp_df_dummy[tmp_df_dummy.flag==0]
	coupons_test_feat = tmp_df_dummy[tmp_df_dummy.flag==1]

	# get the values for the pairwise_distances method
	df_top_10_feat = (coupons_train_feat[coupons_train_feat.coupon_id_hash.isin(top10)]
	    .reset_index()
	    .drop(['flag','coupon_id_hash','index'], axis=1)
	    )
	coupons_test_feat = (coupons_test_feat
		.drop(['flag','coupon_id_hash'], axis=1)
		.values)

	# cosine distance
	dist_mtx = pairwise_distances(coupons_test_feat, df_top_10_feat, metric='cosine')

	# test coupons average distance to most popular train coupons
	mean_distances = np.apply_along_axis(np.mean, 1, dist_mtx)
	df_test_popularity = pd.DataFrame({'coupon_id_hash': coupons_test_ids,
	    'popularity': 1-mean_distances})

	return df_test_popularity


def build_interactions_dictionary(interest_path, test_purchases_path,
	test_visits_path, is_hot=True):

	# interest dataframe
	df_interest = pd.read_pickle(interest_path)
	train_users = df_interest.user_id_hash.unique()
	del(df_interest)

	# test activities
	df_purchases_test = pd.read_pickle(test_purchases_path)
	df_visits_test = pd.read_pickle(test_visits_path)
	df_visits_test.rename(index=str, columns={'view_coupon_id_hash': 'coupon_id_hash'}, inplace=True)

	if is_hot:
		df_vte = df_visits_test[df_visits_test.user_id_hash.isin(train_users)]
		df_pte = df_purchases_test[df_purchases_test.user_id_hash.isin(train_users)]
	else:
		df_vte = df_visits_test[~df_visits_test.user_id_hash.isin(train_users)]
		df_pte = df_purchases_test[~df_purchases_test.user_id_hash.isin(train_users)]

	# dictionary of interactions to evaluate
	id_cols = ['user_id_hash', 'coupon_id_hash']

	df_interactions_test = pd.concat([df_pte[id_cols], df_vte[id_cols]], ignore_index=True)
	df_interactions_test = (df_interactions_test.groupby('user_id_hash')
	    .agg({'coupon_id_hash': 'unique'})
	    .reset_index())
	interactions_test_dict = pd.Series(df_interactions_test.coupon_id_hash.values,
	    index=df_interactions_test.user_id_hash).to_dict()

	return interactions_test_dict


def build_recomendations_dictionary(ranking_df, ranking_metric='interest'):

	df_ranked = ranking_df.sort_values(['user_id_hash', ranking_metric], ascending=[False, False])
	df_ranked = (df_ranked
	    .groupby('user_id_hash')['coupon_id_hash']
	    .apply(list)
	    .reset_index())
	recomendations_dict = pd.Series(df_ranked.coupon_id_hash.values,
	    index=df_ranked.user_id_hash).to_dict()

	return recomendations_dict


def most_popular_recommendations(train_coupons_path, test_coupons_path, interest_path):

	# test coupons popularity
	df_test_popularity = test_coupon_populatiry(train_coupons_path, test_coupons_path)

	# list of purchases and visits for new users
	interactions_test_dict = build_interactions_dictionary(interest_path,
		test_purchases_path, test_visits_path, is_hot=False)

	# ranking dataframe
	left = pd.DataFrame({'user_id_hash':list(interactions_test_dict.keys())})
	left['key'] = 0
	right = pd.read_pickle(test_coupons_path)[['coupon_id_hash']]
	right['key'] = 0
	df_test = (pd.merge(left, right, on='key', how='outer')
	    .drop('key', axis=1))
	df_test = pd.merge(df_test, df_test_popularity, on='coupon_id_hash')

	recomendations_dict = build_recomendations_dictionary(df_test, ranking_metric='popularity')

	return recomendations_dict


def build_lightgbm_train_set(train_coupons_path, train_users_path, interest_path):

	# train coupon features
	df_coupons_train_feat = pd.read_pickle(train_coupons_path)
	drop_cols = [c for c in df_coupons_train_feat.columns
	    if ((not c.endswith('_cat')) or ('method2' in c)) and (c!='coupon_id_hash')]
	df_coupons_train_cat_feat = df_coupons_train_feat.drop(drop_cols, axis=1)

	# train user features
	df_users_train_feat = pd.read_pickle(train_users_path)

	# interest dataframe
	df_interest = pd.read_pickle(interest_path)
	train_users = df_interest.user_id_hash.unique()

	df_train = pd.merge(df_interest, df_users_train_feat, on='user_id_hash')
	df_train = pd.merge(df_train, df_coupons_train_cat_feat, on = 'coupon_id_hash')

	# for the time being we ignore recency
	df_train.drop(['user_id_hash','coupon_id_hash','recency_factor'], axis=1, inplace=True)
	train = df_train.drop('interest', axis=1)
	y_train = df_train.interest
	all_cols = train.columns.tolist()
	cat_cols = [c for c in train.columns if c.endswith("_cat")]

	return train.values, y_train, all_cols, cat_cols, drop_cols


def build_lightgbm_test_set(train_users_path, test_coupons_path,
	test_purchases_path, test_visits_path,
	interest_path, drop_cols):

	interactions_test_dict = build_interactions_dictionary(
		interest_path, test_purchases_path, test_visits_path, is_hot=True)

	df_users_train_feat = pd.read_pickle(train_users_path)
	df_coupons_test_feat = (pd.read_pickle(test_coupons_path)
		.drop(drop_cols, axis=1))

	left = pd.DataFrame({'user_id_hash':list(interactions_test_dict.keys())})
	left['key'] = 0
	right = pd.read_pickle(test_coupons_path)[['coupon_id_hash']]
	right['key'] = 0
	df_test = (pd.merge(left, right, on='key', how='outer')
	    .drop('key', axis=1))
	df_test = pd.merge(df_test, df_users_train_feat, on='user_id_hash')
	df_test = pd.merge(df_test, df_coupons_test_feat, on = 'coupon_id_hash')
	X_test = (df_test
	    .drop(['user_id_hash','coupon_id_hash'], axis=1)
	    .values)
	df_rank = df_test[['user_id_hash','coupon_id_hash']]

	return X_test, df_rank


def compute_mapk(interactions_dict, recomendations_dict):
	actual = []
	pred = []
	for k,_ in recomendations_dict_hot.items():
		actual.append(list(interactions_dict[k]))
		pred.append(list(recomendations_dict[k]))
	return mapk(actual,pred)


if __name__ == '__main__':

	parser = argparse.ArgumentParser(description="full recommender solution combination of most popular and lightgbm")

	parser.add_argument("--inp_dir", type=str, default="../datasets/Ponpare/data_processed")
	parser.add_argument("--train_dir", type=str, default="ftrain")
	parser.add_argument("--test_dir", type=str, default="test")
	parser.add_argument("--model_dir", type=str, default="models")
	args = parser.parse_args()

	# inp_dir = "../datasets/Ponpare/data_processed/"
	# train_dir = "ftrain"
	# test_dir = "test"
	# model_dir = "models"

	inp_dir = args.inp_dir
	train_dir = args.train_dir
	test_dir = args.test_dir
	model_dir = args.model_dir

	train_visits_path = os.path.join(inp_dir,train_dir, 'df_visits_train.p')
	train_purchases_path = os.path.join(inp_dir,train_dir, 'df_purchases_train.p')
	train_coupons_path = os.path.join(inp_dir,train_dir, 'df_coupons_train_feat.p')
	train_users_path = os.path.join(inp_dir,train_dir, 'df_users_train_feat.p')
	test_visits_path = os.path.join(inp_dir,test_dir, 'df_visits_test.p')
	test_purchases_path = os.path.join(inp_dir,test_dir, 'df_purchases_test.p')
	test_coupons_path = os.path.join(inp_dir,test_dir, 'df_coupons_test_feat.p')
	test_users_path = os.path.join(inp_dir,test_dir, 'df_users_test_feat.p')
	interest_path = os.path.join(inp_dir,train_dir, 'df_interest.p')
	best_params_path = os.path.join(inp_dir, model_dir, 'gbm_optimal_parameters.p')

	recomendations_dict_cold = most_popular_recommendations(train_coupons_path,
		test_coupons_path, interest_path)

	train,y_train,all_cols,cat_cols,drop_cols = build_lightgbm_train_set(
		train_coupons_path,
		train_users_path,
		interest_path)
	X_test, df_rank = build_lightgbm_test_set(
		train_users_path,
		test_coupons_path,
		test_purchases_path,
		test_visits_path,
		interest_path,
		drop_cols)

	best = pickle.load(open(best_params_path, "rb"))
	model = lgb.LGBMRegressor(**best)
	model.fit(train,y_train,feature_name=all_cols,categorical_feature=cat_cols)
	preds = model.predict(X_test)
	df_rank['interest'] = preds
	recomendations_dict_hot = build_recomendations_dictionary(df_rank)
	recomendations_dict = recomendations_dict_cold.copy()
	recomendations_dict.update(recomendations_dict_hot)

	interactions_dict_cold = build_interactions_dictionary(
		interest_path,
		test_purchases_path,
		test_visits_path,
		is_hot=False)
	interactions_dict_hot = build_interactions_dictionary(
		interest_path,
		test_purchases_path,
		test_visits_path,
		is_hot=True)
	interactions_dict = interactions_dict_cold.copy()
	interactions_dict.update(interactions_dict_hot)

	final_mapk = compute_mapk(interactions_dict, recomendations_dict)
	print(final_mapk)

