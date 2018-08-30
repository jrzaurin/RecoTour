import numpy as np
import lightgbm as lgb
import multiprocessing
import warnings

from hyperopt import hp, tpe
from hyperopt.fmin import fmin
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import cross_val_score, StratifiedKFold

warnings.filterwarnings("ignore")
cores = multiprocessing.cpu_count()


def lgb_objective(params, lgbdata, additional_params):

	lgb_objective.i+=1

	if 'max_depth' in params.keys(): params['max_depth'] = int(params['max_depth'])
	if 'num_leaves' in params.keys(): params['num_leaves'] = int(params['num_leaves'])

	params.update(additional_params)

	cv_result = lgb.cv(
		params,
		lgbdata,
		nfold=3,
		metrics='rmse',
		num_boost_round=params['n_estimators'],
		early_stopping_rounds=20,
		stratified=False,
		)

	error = cv_result['rmse-mean'][-1]
	print("INFO: iteration {} error {:.3f}".format(lgb_objective.i, error))

	return error


class LGBOptimizer(object):
	def __init__(self, model, feature_names, categorical_feat, metric="rmse",n_evals=5):

		self.params = model.get_params()
		self.n_evals = n_evals
		self.params['verbose'] = -1
		self.feature_names = feature_names
		self.categorical_feat = categorical_feat
		self.metric = metric


	def hypersearch(self, params, **kwargs):

		# set additonal parameters which are the parameters already tunned and
		# removing those that are going to be tunned
		add_params = self.params.copy()
		for k,v in params.items():
			add_params.pop(k)

		# overwrite if something is passed as kwargs (only applicable to
		# add_params)
		for k,v in params.items():
			if k in kwargs.keys(): add_params[k] = kwargs[k]

		partial_objective = lambda params: lgb_objective(
			params,
			self.data,
			additional_params=add_params
			)

		lgb_objective.i=0
		best = fmin(fn=partial_objective,
		            space=params,
		            algo=tpe.suggest,
		            max_evals=self.n_evals)

		return best


	def num_estimators(self, X, y, n_folds=3, learning_rate=0.3, n_estimators=500, early_stop = 20, **kwargs):
		"""
		Setting up a hight learning rate and using early stopping to find the
		number of estimators.
		"""

		params = {}
		params['learning_rate']=learning_rate

		# overwrite if something is passed as kwargs
		for k,v in params.items():
			if k in kwargs.keys(): params[k] = kwargs[k]

		params['verbose'] = -1
		dtrain = lgb.Dataset(X, label=y,
			feature_name=self.feature_names,
			categorical_feature=self.categorical_feat,
			free_raw_data=False)
		self.data = dtrain
		cv_result = lgb.cv(
			params,
			dtrain,
			nfold=3,
			metrics=self.metric,
			num_boost_round=n_estimators,
			early_stopping_rounds=20,
			stratified=False)
		self.params['n_estimators'] = len(cv_result[self.metric+'-mean'])
		self.params['learning_rate'] = learning_rate

		return self


	def depth_and_child_weight(self, X=None, y=None, cw=[1, 50, 2], nl=[30,1024,5], **kwargs):

		if ( (not hasattr(self, 'data')) and (X is None) ) :
			raise AttributeError(" 'LGBOptimizer' object has no attribute 'data' and no X and y arrays are passed")
		assert np.max(nl) <= 1024, "Number of leafs is {}. Only values of 1024 or less are allowed in this implemenation".format(np.max(d))

		if ( (not hasattr(self, 'data')) and (X is not None) ):
			dtrain = lgb.Dataset(X, label=y,
				feature_name=self.feature_names,
				categorical_feature=self.categorical_feat,
				free_raw_data=False)
			self.data = dtrain

		param_space = {
		    'min_child_weight': hp.quniform('min_child_weight', cw[0], cw[1], cw[2]),
		    'num_leaves': hp.quniform('num_leaves', nl[0],nl[1],nl[2])
		}

		best = self.hypersearch(param_space, **kwargs)

		# update tunned parameters
		for k,v in self.params.items():
			if k in best.keys():
				self.params[k] = int(best[k])
		return self


	def min_split_gain(self, X=None, y=None, sl=[0,10], **kwargs):

		if ( (not hasattr(self, 'data')) and (X is None) ) :
			raise AttributeError(" 'LGBOptimizer' object has no attribute 'data' and no X and y arrays are passed")

		if ( (not hasattr(self, 'data')) and (X is not None) ):
			dtrain = lgb.Dataset(X, label=y,
				feature_name=self.feature_names,
				categorical_feature=self.categorical_feat,
				free_raw_data=False)
			self.data = dtrain

		param_space = {'min_split_gain': hp.uniform('min_split_gain', sl[0],sl[1])}

		best = self.hypersearch(param_space, **kwargs)

		for k,v in self.params.items():
			if k in best.keys():
				self.params[k] = best[k]
		return self


	def sample_parameters(self, X=None, y=None, csbt=[0.5, 1.], subs=[0.5, 1.], **kwargs):

		if ( (not hasattr(self, 'data')) and (X is None) ) :
			raise AttributeError(" 'LGBOptimizer' object has no attribute 'data' and no X and y arrays are passed")

		if ( (not hasattr(self, 'data')) and (X is not None) ):
			dtrain = lgb.Dataset(X, label=y,
				feature_name=self.feature_names,
				categorical_feature=self.categorical_feat,
				free_raw_data=False)
			self.data = dtrain

		param_space = {}
		param_space['colsample_bytree'] = hp.uniform('colsample_bytree', csbt[0], csbt[1] )
		param_space['subsample'] = hp.uniform('subsample', csbt[0], csbt[1] )

		best = self.hypersearch(param_space, **kwargs)

		for k,v in self.params.items():
			if k in best.keys():
				self.params[k] = best[k]
		return self


	def regularization(self, X=None, y=None, alpha=[0.01, 1.], gamma=[ 0.01, 1.], **kwargs):

		if ( (not hasattr(self, 'data')) and (X is None) ) :
			raise AttributeError(" 'LGBOptimizer' object has no attribute 'data' and no X and y arrays are passed")

		if ( (not hasattr(self, 'data')) and (X is not None) ):
			dtrain = lgb.Dataset(X, label=y,
				feature_name=self.feature_names,
				categorical_feature=self.categorical_feat,
				free_raw_data=False)
			self.data = dtrain

		param_space = {}
		param_space['reg_alpha'] = hp.uniform('reg_alpha', alpha[0],alpha[1])
		param_space['reg_lambda'] = hp.uniform('reg_lambda',alpha[0],alpha[1])

		best = self.hypersearch(param_space, **kwargs)

		for k,v in self.params.items():
			if k in best.keys():
				self.params[k] = best[k]
		return self


	def fine_tunning(self, n_folds = 3, lr_r=np.arange(0.005,0.3,0.005),
		num_boost_round=500, early_stop=20, **kwargs):

		if not hasattr(self, 'data'):
			raise AttributeError(" 'LGBOptimizer' object has no attribute 'data'")

		params = dict(self.params)

		# overwrite if something is passed as kwargs
		for k,v in params.items():
			if k in kwargs.keys(): params[k] = kwargs[k]

		partial_cv = lambda param_lr: lgb.cv(
	        param_lr,
	        self.data,
			num_boost_round = num_boost_round,
	        metrics=self.metric,
	        nfold=n_folds,
	        stratified=False,
	        early_stopping_rounds=early_stop)

		results = {}
		for i, lr in enumerate(lr_r):
			params['learning_rate'] = lr
			cv_result = partial_cv(params)

			n_est = len(cv_result[self.metric+'-mean'])
			error = cv_result[self.metric+'-mean'][-1]
			results[i] = (lr, n_est, error)
			print("INFO: iteration {} of {}. learning rate: {}. estimators: {}. Error {}: {}".format(i+1, len(lr_r)+1, round(lr,4), n_est, self.metric, round(error, 4)))

		best_idx = np.argmin([v[2] for v in list(results.values())])
		best = results[best_idx]
		self.params['learning_rate'] = best[0]
		self.params['n_estimators'] = best[1]

		return self

	def full_optimization(self, X, y, verbose=True):

		if verbose: print('INFO: fixing a high learning rate and tunning the number of estimators')
		self.num_estimators(X,y)

		if verbose: print('INFO: tunning tree specific parameters: max_depth and min_child_weight')
		self.depth_and_child_weight()

		if verbose: print('INFO: tunning loss reduction')
		self.min_split_gain()

		if verbose: print('INFO: tunning sampling parameters: subsample and colsample_bytree')
		self.sample_parameters()

		if verbose: print('INFO: tunning regularization')
		self.regularization()

		if verbose: print('INFO: Fine tunning learning rate')
		self.fine_tunning()

		return self
