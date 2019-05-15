import numpy as np
import pandas as pd
import argparse

from pathlib import Path
from time import time
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split


def standard_split(df, data_path):

	# Cardinality
	n_user = df.user.nunique()
	n_item = df.item.nunique()
	n_rank = df['rank'].nunique()

	# train/test split
	features = ['user', 'item', 'rank']
	target = ['rating']
	X = df[features].values.astype(np.int32)
	y = df[target].values.astype(np.float32)
	train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=1)

	# Save
	np.savez(data_path/"standard_split.npz", train_x=train_x, train_y=train_y,
		test_x=test_x, test_y=test_y, n_user=n_user, n_item=n_item,
		n_ranks=n_rank)


def neuralcf_split(df, data_path):
	# Xiangnan He, et al, 2017 train/test split with implicit negative feedback

	# sort by rank
	dfc = df.copy()

	# Cardinality
	n_user = dfc.user.nunique()
	n_item = dfc.item.nunique()

	dfc.sort_values(['user','rank'], ascending=[True,True], inplace=True)
	dfc.reset_index(inplace=True, drop=True)

	# use last ratings for testing and all the previous for training
	test = dfc.groupby('user').tail(1)
	train = pd.merge(dfc, test, on=['user','item'],
		how='outer', suffixes=('', '_y'))
	train = train[train.rating_y.isnull()]
	test = test[['user','item','rating', 'rank']]
	train = train[['user','item','rating', 'rank']]

	# select 99 random movies per user that were never rated
	all_items = dfc.item.unique()
	negative = (dfc.groupby("user")['item']
	    .apply(list)
	    .reset_index()
	    )
	rated_items = negative.item.tolist()

	def sample_not_rated(item_list, rseed=1, n=99):
		np.random.seed=rseed
		return np.random.choice(np.setdiff1d(all_items, item_list), n)

	print("sampling not rated items for negative feedback...")
	start = time()
	non_rated_items = Parallel(n_jobs=4)(delayed(sample_not_rated)(ri) for ri in rated_items)
	end = time() - start
	print("sampling for negative feedback took {} min".format(round(end/60,2)))

	# manipulating the df to look like this:
	# positive  item_n0  item_n1  item_n2  item_n3  item_n4  item_n5  item_n6  ...
	# (0, 522)    27984    22902    28875    35434    28240    32183    46373  ...
	negative['negative'] = non_rated_items
	negative.drop('item', axis=1, inplace=True)
	negative= test.merge(negative, on='user')
	negative['positive'] = negative[['user', 'item']].apply(tuple, axis=1)
	negative.drop(['user','item', 'rating', 'rank'], axis=1, inplace=True)
	negative = negative[['positive','negative']]
	negative[['item_n'+str(i) for i in range(99)]] = \
		pd.DataFrame(negative.negative.values.tolist(), index= negative.index)
	negative.drop('negative', axis=1, inplace=True)

	features = ['user', 'item', 'rank']
	target = ['rating']
	train_x = train[features].values.astype(np.int32)
	train_y = train[target].values.astype(np.float32)
	test_x = test[features].values.astype(np.int32)
	test_y = test[target].values.astype(np.float32)

	# Save
	np.savez(data_path/"neuralcf_split.npz", train_x=train_x, train_y=train_y,
		test_x=test_x, test_y=test_y, negative=negative.values, n_user=n_user,
		n_item=n_item)


def prepare_amazon(data_path, input_fname):

	df = pd.read_json(data_path/input_fname, lines=True)

	keep_cols = ['reviewerID', 'asin', 'unixReviewTime', 'overall']
	new_colnames = ['user', 'item', 'timestamp', 'rating']
	df = df[keep_cols]
	df.columns = new_colnames

	# rank of items bought
	df['rank'] = df.groupby("user")["timestamp"].rank(ascending=True, method='dense')
	df.drop("timestamp", axis=1, inplace=True)

	# mapping user and item ids to integers
	user_mappings = {k:v for v,k in enumerate(df.user.unique())}
	item_mappings = {k:v for v,k in enumerate(df.item.unique())}
	df['user'] = df['user'].map(user_mappings)
	df['item'] = df['item'].map(item_mappings)

	standard_split(df, data_path)
	neuralcf_split(df, data_path)


if __name__ == '__main__':

	parser = argparse.ArgumentParser(description="prepare Amazon dataset")

	parser.add_argument("--input_dir",type=str, default="/home/ubuntu/projects/RecoTour/datasets/Amazon")
	parser.add_argument("--input_fname",type=str, default="reviews_Movies_and_TV_5.json")
	args = parser.parse_args()

	DATA_PATH = Path(args.input_dir)
	reviews = args.input_fname

	prepare_amazon(DATA_PATH, reviews)
