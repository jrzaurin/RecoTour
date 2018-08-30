import numpy as np
import pandas as pd
import pickle
import os
import argparse
import lightgbm as lgb
import warnings
import multiprocessing

from recolearn.metrics.average_precision import mapk
from sklearn.metrics.pairwise import pairwise_distances
from hyperopt import hp, tpe, fmin, Trials

warnings.filterwarnings("ignore")
cores = multiprocessing.cpu_count()


class Interactions(object):
	def __init__(self, is_hot):
		self.is_hot = is_hot

	def interactions_dictionary(self, df_purchases_eval, df_visits_eval, df_metric):

		train_users = df_metric.user_id_hash.unique()

		if self.is_hot:
			df_vte = df_visits_eval[df_visits_eval.user_id_hash.isin(train_users)]
			df_pte = df_purchases_eval[df_purchases_eval.user_id_hash.isin(train_users)]
		else:
			df_vte = df_visits_eval[~df_visits_eval.user_id_hash.isin(train_users)]
			df_pte = df_purchases_eval[~df_purchases_eval.user_id_hash.isin(train_users)]

		id_cols = ['user_id_hash', 'coupon_id_hash']

		df_interactions_test = pd.concat([df_pte[id_cols], df_vte[id_cols]], ignore_index=True)
		df_interactions_test = (df_interactions_test.groupby('user_id_hash')
		    .agg({'coupon_id_hash': 'unique'})
		    .reset_index())
		interactions_dict = pd.Series(df_interactions_test.coupon_id_hash.values,
		    index=df_interactions_test.user_id_hash).to_dict()

		return interactions_dict


	def recomendations_dictionary(self, df, ranking_metric):

		df_ranked = df.sort_values(['user_id_hash', ranking_metric], ascending=[False, False])
		df_ranked = (df_ranked
		    .groupby('user_id_hash')['coupon_id_hash']
		    .apply(list)
		    .reset_index())
		recomendations_dict = pd.Series(df_ranked.coupon_id_hash.values,
		    index=df_ranked.user_id_hash).to_dict()

		return recomendations_dict


class MPRec(Interactions):

	def __init__(self, work_dir, train_dir):
		super(MPRec, self).__init__(is_hot=False)

		self.work_dir = work_dir
		self.train_dir = train_dir
		self.train_path = None

	def set_experiment(self):

		self.train_path = os.path.join(self.work_dir, self.train_dir)
		self.test_path = os.path.join(self.work_dir, 'test')

		self.df_coupons_train_feat = pd.read_pickle(os.path.join(self.train_path, 'df_coupons_train_feat.p'))
		self.df_purchases_train = pd.read_pickle(os.path.join(self.train_path, 'df_purchases_train.p'))
		self.df_visits_train = pd.read_pickle(os.path.join(self.train_path, 'df_visits_train.p'))
		self.df_visits_train.rename(index=str, columns={'view_coupon_id_hash': 'coupon_id_hash'}, inplace=True)

		self.df_coupons_test_feat = pd.read_pickle(os.path.join(self.test_path, 'df_coupons_test_feat.p'))
		self.df_purchases_test = pd.read_pickle(os.path.join(self.test_path, 'df_purchases_test.p'))
		self.df_visits_test = pd.read_pickle(os.path.join(self.test_path, 'df_visits_test.p'))
		self.df_visits_test.rename(index=str, columns={'view_coupon_id_hash': 'coupon_id_hash'}, inplace=True)

		self.df_interest =  pd.read_pickle(os.path.join(self.train_path, 'df_interest.p'))


	def most_popular_recommendations(self):

		if not self.train_path:
			self.set_experiment()

		# test coupons popularity
		df_test_popularity = self.test_coupons_popularity()

		# list of purchases and visits for new users
		cold_interactions_dict = self.interactions_dictionary(
			self.df_purchases_test,
			self.df_visits_test,
			self.df_interest)

		# ranking dataframe
		left = pd.DataFrame({'user_id_hash':list(cold_interactions_dict.keys())})
		left['key'] = 0
		right = self.df_coupons_test_feat[['coupon_id_hash']]
		right['key'] = 0
		df_test = (pd.merge(left, right, on='key', how='outer')
		    .drop('key', axis=1))
		df_test = pd.merge(df_test, df_test_popularity, on='coupon_id_hash')

		rec_dict = self.recomendations_dictionary(df_test, ranking_metric='popularity')

		return rec_dict


	def test_coupons_popularity(self):

		if not self.train_path:
			self.set_experiment()

		topN = self.topn_train_coupon_populatiry()

		coupons_test_ids = self.df_coupons_test_feat.coupon_id_hash.values

		id_cols = ['coupon_id_hash']
		cat_cols = [c for c in self.df_coupons_train_feat.columns if c.endswith('_cat')]
		num_cols = [c for c in self.df_coupons_train_feat.columns if
			(c not in cat_cols) and (c not in id_cols)]

		# Compute distances
		self.df_coupons_train_feat['flag'] = 0
		self.df_coupons_test_feat['flag'] = 1

		tmp_df = pd.concat(
			[self.df_coupons_train_feat,self.df_coupons_test_feat],
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
		df_top_N_feat = (coupons_train_feat[coupons_train_feat.coupon_id_hash.isin(topN)]
		    .reset_index()
		    .drop(['flag','coupon_id_hash','index'], axis=1)
		    )
		coupons_test_feat = (coupons_test_feat
			.drop(['flag','coupon_id_hash'], axis=1)
			.values)

		# cosine distance
		dist_mtx = pairwise_distances(coupons_test_feat, df_top_N_feat, metric='cosine')

		# test coupons average distance to most popular train coupons
		mean_distances = np.apply_along_axis(np.mean, 1, dist_mtx)
		df_test_popularity = pd.DataFrame({'coupon_id_hash': coupons_test_ids,
		    'popularity': 1-mean_distances})

		return df_test_popularity


	def topn_train_coupon_populatiry(self, n=10):

		if not self.train_dir:
			self.set_experiment()

		# popularity = n_purchases + 0.1*n_visits
		df_n_purchases = (self.df_purchases_train
			.coupon_id_hash
			.value_counts()
			.reset_index())
		df_n_purchases.columns = ['coupon_id_hash','counts']
		df_n_visits = (self.df_visits_train
			.coupon_id_hash
			.value_counts()
			.reset_index())
		df_n_visits.columns = ['coupon_id_hash','counts']

		df_popularity = df_n_purchases.merge(df_n_visits, on='coupon_id_hash', how='left')
		df_popularity.fillna(0, inplace=True)
		df_popularity['popularity'] = df_popularity['counts_x'] + 0.1*df_popularity['counts_y']
		df_popularity.sort_values('popularity', ascending=False , inplace=True)

		# select top N most popular coupons from the training dataset
		topN = df_popularity.coupon_id_hash.tolist()[:n]

		return topN


class LGBDataprep(Interactions):
	def __init__(self):
		super(LGBDataprep, self).__init__(is_hot=True)


	def build_lightgbm_train_set(self, df_coupons_train_feat,
		df_users_train_feat, df_interest):

		drop_cols = [c for c in df_coupons_train_feat.columns
		    if ((not c.endswith('_cat')) or ('method2' in c)) and (c!='coupon_id_hash')]
		df_coupons_train_cat_feat = df_coupons_train_feat.drop(drop_cols, axis=1)

		train_users = df_interest.user_id_hash.unique()

		df_train = pd.merge(df_interest, df_users_train_feat, on='user_id_hash')
		df_train = pd.merge(df_train, df_coupons_train_cat_feat, on = 'coupon_id_hash')

		# for the time being we ignore recency
		df_train.drop(['user_id_hash','coupon_id_hash','recency_factor'], axis=1, inplace=True)
		train = df_train.drop('interest', axis=1)
		y_train = df_train.interest
		all_cols = train.columns.tolist()
		cat_cols = [c for c in train.columns if c.endswith("_cat")]

		lgtrain = lgb.Dataset(train,
			label=y_train,
			feature_name=all_cols,
			categorical_feature = cat_cols,
			free_raw_data=False)

		return lgtrain, all_cols, cat_cols, drop_cols


	def build_lightgbm_test_set(self, df_coupons_test_feat, df_users_train_feat,
		df_purchases_test, df_visits_test, df_interest, drop_cols):

		hot_interactions_dict = self.interactions_dictionary(
			df_purchases_test,
			df_visits_test,
			df_interest)

		df_coupons_test_feat_dc = df_coupons_test_feat.drop(drop_cols, axis=1)

		left = pd.DataFrame({'user_id_hash':list(hot_interactions_dict.keys())})
		left['key'] = 0
		right = df_coupons_test_feat[['coupon_id_hash']]
		right['key'] = 0
		df_test = (pd.merge(left, right, on='key', how='outer')
		    .drop('key', axis=1))
		df_test = pd.merge(df_test, df_users_train_feat, on='user_id_hash')
		df_test = pd.merge(df_test, df_coupons_test_feat_dc, on = 'coupon_id_hash')
		X_test = (df_test
		    .drop(['user_id_hash','coupon_id_hash'], axis=1)
		    .values)

		df_to_rank = df_test[['user_id_hash','coupon_id_hash']]

		return X_test, df_to_rank


class LGBOptimize(LGBDataprep):
	def __init__(self, work_dir):
		super(LGBDataprep, self).__init__(is_hot=True)

		self.work_dir = work_dir

		optimal_param_path = os.path.join(self.work_dir, 'models', 'gbm_optimal_parameters.p')
		if os.path.isfile(optimal_param_path):
			self.best = pickle.load(open(optimal_param_path, "rb"))
		else:
			self.best = None

		self.train_path=None


	def set_experiment(self):

		self.train_path = os.path.join(self.work_dir, 'train')
		self.test_path = os.path.join(self.work_dir, 'valid')
		self.df_coupons_test_feat = pd.read_pickle(os.path.join(self.test_path, 'df_coupons_valid_feat.p'))
		self.df_purchases_test = pd.read_pickle(os.path.join(self.test_path, 'df_purchases_valid.p'))
		self.df_visits_test = pd.read_pickle(os.path.join(self.test_path, 'df_visits_valid.p'))
		self.df_visits_test.rename(index=str, columns={'view_coupon_id_hash': 'coupon_id_hash'}, inplace=True)


		self.df_coupons_train_feat = pd.read_pickle(os.path.join(self.train_path, 'df_coupons_train_feat.p'))
		self.df_users_train_feat = pd.read_pickle(os.path.join(self.train_path, 'df_users_train_feat.p'))
		self.df_purchases_train = pd.read_pickle(os.path.join(self.train_path, 'df_purchases_train.p'))
		self.df_visits_train = pd.read_pickle(os.path.join(self.train_path, 'df_visits_train.p'))
		self.df_visits_train.rename(index=str, columns={'view_coupon_id_hash': 'coupon_id_hash'}, inplace=True)

		self.df_interest =  pd.read_pickle(os.path.join(self.train_path, 'df_interest.p'))


	def optimize(self, maxevals=50):
		print("INFO: optimising lightgbm")

		if not self.train_path:
			self.set_experiment()

		train, all_cols, cat_cols, drop_cols = self.build_lightgbm_train_set(
			self.df_coupons_train_feat,
			self.df_users_train_feat,
			self.df_interest)
		X_valid, df_eval = self.build_lightgbm_test_set(
			self.df_coupons_test_feat,
			self.df_users_train_feat,
			self.df_purchases_test,
			self.df_visits_test,
			self.df_interest,
			drop_cols)
		interactions_valid_dict = self.interactions_dictionary(
			self.df_purchases_test,
			self.df_visits_test,
			self.df_interest)
		early_stop_dict = {}

		objective = self.get_objective(train, all_cols, cat_cols, X_valid, df_eval, interactions_valid_dict, early_stop_dict)
		param_space = self.hyperparameter_space()

		trials = Trials()
		objective.i = 0
		best = fmin(fn=objective,
		            space=param_space,
		            algo=tpe.suggest,
		            max_evals=maxevals,
		            trials=trials)

		best['num_boost_round'] = early_stop_dict[trials.best_trial['tid']]
		best['num_leaves'] = int(best['num_leaves'])
		best['verbose'] = -1

		if not self.best:
			self.best = best

		pickle.dump(best, open(os.path.join(self.work_dir, 'models', 'gbm_optimal_parameters.p'),'wb') )


	def get_objective(self, train, all_cols, cat_cols, X_valid, df_eval, interactions_valid_dict, early_stop_dict, verbose=True):

		def objective(params):
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
			train,
			nfold=3,
			metrics='rmse',
			num_boost_round=params['num_boost_round'],
			early_stopping_rounds=20,
			stratified=False,
			)
			early_stop_dict[objective.i] = len(cv_result['rmse-mean'])
			params['num_boost_round'] = len(cv_result['rmse-mean'])

			model = lgb.LGBMRegressor(**params)
			model.fit(train.data,train.label,feature_name=all_cols,categorical_feature=cat_cols)
			preds = model.predict(X_valid)

			df_eval['interest'] = preds
			recomendations_dict = self.recomendations_dictionary(df_eval, ranking_metric='interest')

			actual = []
			pred = []
			for k,_ in recomendations_dict.items():
				actual.append(list(interactions_valid_dict[k]))
				pred.append(list(recomendations_dict[k]))

			result = mapk(actual,pred)

			if verbose:
				print("INFO: iteration {} MAP {:.3f}".format(objective.i, result))

			objective.i+=1

			return 1-result

		return objective


	def hyperparameter_space(self, param_space=None):

		space = {
			'learning_rate': hp.uniform('learning_rate', 0.01, 0.5),
			'num_boost_round': hp.quniform('num_boost_round', 50, 500, 20),
			'num_leaves': hp.quniform('num_leaves', 30,1024,5),
		    'min_child_weight': hp.quniform('min_child_weight', 1, 50, 2),
		    'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1.),
		    'subsample': hp.uniform('subsample', 0.5, 1.),
		    'reg_alpha': hp.uniform('reg_alpha', 0.01, 1.),
		    'reg_lambda': hp.uniform('reg_lambda', 0.01, 1.),
		}

		if param_space:
			return param_space
		else:
			return space


class LGBRec(LGBDataprep):
	def __init__(self, work_dir, train_dir):
		super(LGBDataprep, self).__init__(is_hot=True)

		self.work_dir = work_dir
		self.train_dir = train_dir

		optimal_param_path = os.path.join(self.work_dir, 'models', 'gbm_optimal_parameters.p')
		if os.path.isfile(optimal_param_path):
			self.best = pickle.load(open(optimal_param_path, "rb"))
		else:
			self.best = None

		self.train_path=None

	def set_experiment(self):

		self.train_path = os.path.join(self.work_dir, self.train_dir)
		self.test_path = os.path.join(self.work_dir, 'test')

		self.df_coupons_test_feat = pd.read_pickle(os.path.join(self.test_path, 'df_coupons_test_feat.p'))
		self.df_purchases_test = pd.read_pickle(os.path.join(self.test_path, 'df_purchases_test.p'))
		self.df_visits_test = pd.read_pickle(os.path.join(self.test_path, 'df_visits_test.p'))
		self.df_visits_test.rename(index=str, columns={'view_coupon_id_hash': 'coupon_id_hash'}, inplace=True)

		self.df_coupons_train_feat = pd.read_pickle(os.path.join(self.train_path, 'df_coupons_train_feat.p'))
		self.df_users_train_feat = pd.read_pickle(os.path.join(self.train_path, 'df_users_train_feat.p'))
		self.df_purchases_train = pd.read_pickle(os.path.join(self.train_path, 'df_purchases_train.p'))
		self.df_visits_train = pd.read_pickle(os.path.join(self.train_path, 'df_visits_train.p'))
		self.df_visits_train.rename(index=str, columns={'view_coupon_id_hash': 'coupon_id_hash'}, inplace=True)

		self.df_interest =  pd.read_pickle(os.path.join(self.train_path, 'df_interest.p'))


	def lightgbm_recommendations(self, best=None):

		if (not best) and (not self.best):
			raise ValueError('Please, provide lightgbm parameters or run optimization first')

		best = best or self.best

		if not self.train_path:
			self.set_experiment()

		train, all_cols, cat_cols, drop_cols = self.build_lightgbm_train_set(
			self.df_coupons_train_feat,
			self.df_users_train_feat,
			self.df_interest)
		X_test, df_test = self.build_lightgbm_test_set(
			self.df_coupons_test_feat,
			self.df_users_train_feat,
			self.df_purchases_test,
			self.df_visits_test,
			self.df_interest,
			drop_cols)

		model = lgb.LGBMRegressor(**best)
		model.fit(train.data,train.label,feature_name=all_cols,categorical_feature=cat_cols)
		preds = model.predict(X_test)
		df_test['interest'] = preds

		rec_dict = self.recomendations_dictionary(df_test, ranking_metric='interest')

		return rec_dict


def compute_mapk(interactions_dict, recomendations_dict):
	actual = []
	pred = []
	for k,_ in recomendations_dict.items():
		actual.append(list(interactions_dict[k]))
		pred.append(list(recomendations_dict[k]))
	return mapk(actual,pred)


def load_interactions_test_data(work_dir, train_dir):
	df_purchases_test = pd.read_pickle(os.path.join(work_dir, 'test', 'df_purchases_test.p'))
	df_visits_test = pd.read_pickle(os.path.join(work_dir, 'test', 'df_visits_test.p'))
	df_visits_test.rename(index=str, columns={'view_coupon_id_hash': 'coupon_id_hash'}, inplace=True)
	df_interest = pd.read_pickle(os.path.join(work_dir, train_dir, 'df_interest.p'))
	return df_purchases_test, df_visits_test, df_interest


if __name__ == '__main__':


	parser = argparse.ArgumentParser(description="full recommender solution combination of most popular and lightgbm")

	parser.add_argument(
		"--root_data_dir",
		type=str, default="/home/ubuntu/projects/RecoTour/datasets/Ponpare/",)
	args = parser.parse_args()

	parser.add_argument("--work_dir",
		type=str, default=args.root_data_dir+"data_processed")
	parser.add_argument("--train_dir",
		type=str, default="train")
	args = parser.parse_args()

	lgbopt = LGBOptimize(args.work_dir)
	best_params = lgbopt.optimize(maxevals=5)

	lgbrec = LGBRec(
		args.work_dir,
		args.train_dir
		)
	hot_recommendations = lgbrec.lightgbm_recommendations()

	mprec = MPRec(
		args.work_dir,
		args.train_dir
		)
	cold_recommendations = mprec.most_popular_recommendations()

	df_purchases_test, df_visits_test, df_interest = load_interactions_test_data(
		args.work_dir,
		args.train_dir
		)
	cold_interactions = Interactions(is_hot=False)
	cold_interactions_dict = cold_interactions.interactions_dictionary(
		df_purchases_test,
		df_visits_test,
		df_interest
		)

	hot_interactions = Interactions(is_hot=True)
	hot_interactions_dict = hot_interactions.interactions_dictionary(
		df_purchases_test,
		df_visits_test,
		df_interest
		)

	interactions_dict = cold_interactions_dict.copy()
	interactions_dict.update(hot_interactions_dict)
	recomendations_dict = cold_recommendations.copy()
	recomendations_dict.update(hot_recommendations)

	final_mapk = compute_mapk(interactions_dict, recomendations_dict)
	print(final_mapk)
