import numpy as np
import pandas as pd
import pickle
import os
import scipy.sparse as sp
import argparse
import csv

from tqdm import tqdm
from pathlib import Path


# For train/test split method 1
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


# For train/test split method 2
def create_user_list(df, n_users):
	"""
	Creates a list of dictionaries where keys are items and values are the
	corresponding timestamp. Users are continuous integers, so list index 0
	corresponds to user_id = 0
	"""
	user_list = [dict() for u in range(n_users)]
	for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
	    # this could be .append()
	    user_list[row['user']][row['item']] = row['timestamp']
	return user_list


def split_train_test(user_list, p=0.8, time_order=False):
	"""
	Train/Test split. This function will take the dictionary from create_user_list to
	perform the initial train/test split or its own output to perform train/valid split
	"""
	train_user_list = [None] * len(user_list)
	test_user_list  = [None] * len(user_list)

	for user, item_list in enumerate(user_list):
	    if time_order:
	        # Choose latest item
	        if isinstance(item_list, dict):
	            items = [i[0] for i in sorted(item_list.items(), key=lambda x: x[1])]
	            test_user_list[user]  = items[-1]
	            train_user_list[user] = items[:-1]
	        elif isinstance(item_list, (list, np.ndarray)):
	            print('user_list is assumbed to be sorted with the most recent item being the last one.')
	            test_user_list[user]  = item_list[-1]
	            train_user_list[user] = item_list[:-1]
	    else:
	        # Random select
	        if isinstance(item_list, dict):
	            items =list(item_list.keys())
	            sz = np.floor(len(items)*p).astype('int')
	            train_user_list[user] = np.random.choice(items, sz, replace=False)
	            test_user_list[user] = np.setdiff1d(items, train_user_list[user])
	        elif isinstance(item_list, (list, np.ndarray)):
	            sz = np.floor(len(item_list)*p).astype('int')
	            train_user_list[user] = np.random.choice(item_list, sz, replace=False)
	            test_user_list[user] = np.setdiff1d(item_list, train_user_list[user])

	return train_user_list, test_user_list


def create_pair(user_list):
	"""
	Create pairs (user,item)
	"""
	pair = []
	for user, item_set in enumerate(user_list):
	    pair.extend([(user, item) for item in item_set])
	return pair


def fill_sp_mtx(dataset, n_users, n_items):
	R = sp.dok_matrix((n_users, n_items), dtype=np.float32)
	for u, itemset in enumerate(dataset):
	    for i in itemset:
	        R[u, i] = 1
	return R.tocsr()


if __name__ == '__main__':

	parser = argparse.ArgumentParser(description="Prepare the Amazon movies dataset.")
	parser.add_argument('--input_dir', type=str,
		default="/Users/javier/ml_exercises_python/RecoTour/Amazon/neural_graph_cf/Data/amazon-movies/",
		help="Dir path for raw data")
	parser.add_argument('--input_data', type=str, default="reviews_Movies_and_TV_5.json.gz",
		help="File name for raw data")
	parser.add_argument('--output_data', type=str, default="amazon_movies.npz",
		help="File name for preprocessed data")
	parser.add_argument('--user_list', type=str, default="user_list.p",
		help="File name for user_list")
	args = parser.parse_args()

	DATA_PATH = Path(args.input_dir)
	reviews = args.input_data
	output_data = args.output_data
	user_list_fname = args.user_list

	print("Reading amazon movies dataset...")
	df = pd.read_json(DATA_PATH/reviews, lines=True)
	keep_cols = ['reviewerID', 'asin', 'unixReviewTime', 'overall']
	new_colnames = ['user', 'item', 'timestamp', 'rating']
	df = df[keep_cols]
	df.columns = new_colnames

	df.sort_values(['user','timestamp'], ascending=[True,True], inplace=True)
	df.reset_index(inplace=True, drop=True)

	user_mappings, item_mappings, dfm = map_user_items(df)

	# METHOD 1. Designed to reproduce Xiang Wang et al. 2019 paper.
	print("creating dataframe with lists of interactions...")
	df1 = dfm[['user', 'item']]
	interactions_df = tolist(df1)

	# 80-20 train/test split
	print("Method 1 split (and save)...")
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

	# METHOD 2. From here: https://github.com/sh0416/bpr/blob/master/preprocess.py
	n_users = len(dfm['user'].unique())
	n_items = len(dfm['item'].unique())

	# this is an expensive process, so if is in disk, load it
	if os.path.isfile(DATA_PATH/user_list_fname):
		print("loading user_list...")
		total_user_list = pickle.load(open(DATA_PATH/user_list_fname, "rb"))
	else:
		print("creating and saving user_list...")
		total_user_list = create_user_list(dfm, n_users)
		pickle.dump(total_user_list, open(DATA_PATH/user_list_fname, "wb"))

	print("Method 2 train/test split...")
	# 80-20 train/test split
	train_user_list, test_user_list = split_train_test(total_user_list)
	# 90-10 train/valid split
	train_user_list, valid_user_list = split_train_test(train_user_list, p=0.9)

	train_pair = create_pair(train_user_list)

	print("Creating sparse matrices and saving to npz format...")
	train_w = fill_sp_mtx(train_user_list, n_users, n_items)
	valid_w = fill_sp_mtx(valid_user_list, n_users, n_items)
	test_w = fill_sp_mtx(test_user_list, n_users, n_items)

	# save
	np.savez(DATA_PATH/output_data, train_w=train_w, valid_w=valid_w, test_w=test_w,
		train_pair=train_pair, n_users=n_users, n_items=n_items)
