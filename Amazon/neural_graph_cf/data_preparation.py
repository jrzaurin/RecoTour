import numpy as np
import pandas as pd
import pickle
import os
import scipy.sparse as sp
import argparse
import csv

from tqdm import tqdm
from pathlib import Path


def map_user_items(df):
	"""
	Function to map users and items into continuous integers
	"""
	dfc = df.copy()
	user_mappings = {k:v for v,k in enumerate(dfc.user.unique())}
	item_mappings = {k:v for v,k in enumerate(dfc.item.unique())}

	# save with " " separated format
	user_list = pd.DataFrame.from_dict(user_mappings, orient='index').reset_index()
	user_list.columns = ['orig_id', 'remap_id']
	item_list = pd.DataFrame.from_dict(item_mappings, orient='index').reset_index()
	item_list.columns = ['orig_id', 'remap_id']
	user_list.to_csv(DATA_PATH/'user_list.txt', sep=" ", index=False)
	item_list.to_csv(DATA_PATH/'item_list.txt', sep=" ", index=False)

	dfc['user'] = dfc['user'].map(user_mappings).astype(np.int64)
	dfc['item'] = dfc['item'].map(item_mappings).astype(np.int64)

	return user_mappings, item_mappings, dfc


def tolist(df):
	"""
	Build a dataframe (user, list of items)
	"""
	keys, values = df.sort_values('user').values.T
	ukeys, index = np.unique(keys, True)
	arrays = np.split(values, index[1:])
	df2 = pd.DataFrame({'user':ukeys, 'item':[list(a) for a in arrays]})
	return df2


def train_test_split(u, i_l, p=0.8):
	s = np.floor(len(i_l)*p).astype('int')
	train = list(np.random.choice(i_l, s, replace=False))
	test  = list(np.setdiff1d(i_l, train))
	return ([u]+train, [u]+test)


if __name__ == '__main__':

	parser = argparse.ArgumentParser(description="Prepare the Amazon movies dataset.")
	parser.add_argument('--input_dir', type=str,
		default="/Users/javier/ml_exercises_python/RecoTour/Amazon/neural_graph_cf/Data/amazon-movies/",
		help="Dir path for raw data")
	parser.add_argument('--input_data', type=str, default="reviews_Movies_and_TV_5.json.gz",
		help="File name for raw data")
	args = parser.parse_args()

	DATA_PATH = Path(args.input_dir)
	reviews = args.input_data
	reviews_csv = 'reviews_Movies_and_TV_5.csv'

	print("Reading amazon movies dataset...")
	df = pd.read_json(DATA_PATH/reviews, lines=True)
	keep_cols = ['reviewerID', 'asin', 'unixReviewTime', 'overall']
	new_colnames = ['user', 'item', 'timestamp', 'rating']
	df = df[keep_cols]
	df.columns = new_colnames
	df.to_csv(DATA_PATH/reviews_csv, index=False)

	df.sort_values(['user','timestamp'], ascending=[True,True], inplace=True)
	df.reset_index(inplace=True, drop=True)

	user_mappings, item_mappings, dfm = map_user_items(df)

	print("creating dataframe with lists of interactions...")
	df1 = dfm[['user', 'item']]
	interactions_df = tolist(df1)

	# 80-20 train/test split
	print("Train/Test split (and save)...")
	interactions_l = [train_test_split(r['user'], r['item']) for i,r in interactions_df.iterrows()]
	train = [interactions_l[i][0] for i in range(len(interactions_l))]
	test =  [interactions_l[i][1] for i in range(len(interactions_l))]

	# 90-10 train/valid split
	tr_interactions_l = [train_test_split(t[0], t[1:], p=0.9) for t in train]
	train = [tr_interactions_l[i][0] for i in range(len(tr_interactions_l))]
	valid = [tr_interactions_l[i][1] for i in range(len(tr_interactions_l))]

	# save
	train_fname = DATA_PATH/'train.txt'
	valid_fname = DATA_PATH/'valid.txt'
	test_fname = DATA_PATH/'test.txt'

	with open(train_fname, 'w') as trf, open(valid_fname, 'w') as vaf, open(test_fname, 'w') as tef:
	    trwrt = csv.writer(trf, delimiter=' ')
	    vawrt = csv.writer(vaf, delimiter=' ')
	    tewrt = csv.writer(tef, delimiter=' ')
	    trwrt.writerows(train)
	    vawrt.writerows(valid)
	    tewrt.writerows(test)