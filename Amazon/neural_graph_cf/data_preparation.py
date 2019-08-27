'''
Data preparation process. I will use two methods. One designed to reproduced
Xiang Wang Neural Graph Collaborative Filtering paper and the other one using
an approach based on Xiangnan He Neural Collaborative Filtering paper
'''
import numpy as np
import pandas as pd
import pickle
import os
import scipy.sparse as sp
import argparse
import csv


from copy import copy
from time import time
from tqdm import tqdm
from pathlib import Path
from joblib import Parallel, delayed
from scipy.sparse import save_npz


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


def array2mtx(interactions):
	# code here is a direct copy and paste from my code here:
	# https://github.com/jrzaurin/RecoTour/blob/master/Amazon/neural_cf/prepare_data.py
    num_users = interactions[:,0].max()
    num_items = interactions[:,1].max()
    mat = sp.dok_matrix((num_users+1, num_items+1), dtype=np.float32)
    for user, item, rating in interactions:
            mat[user, item] = rating
    return mat.tocsr()


def neuralcf_split(df, data_path):
	# Xiangnan He, et al, 2017 train/test split with implicit negative feedback
	# code here is a direct copy and paste from my code here:
	# https://github.com/jrzaurin/RecoTour/blob/master/Amazon/neural_cf/prepare_data.py

	dfc = df.copy()

	# Cardinality
	n_users = df.user.nunique()
	n_items = df.item.nunique()

	# sort by rank
	dfc.sort_values(['user','rank'], ascending=[True,True], inplace=True)
	dfc.reset_index(inplace=True, drop=True)

	# use last ratings for testing and all the previous for training
	test = dfc.groupby('user').tail(1)
	train = pd.merge(dfc, test, on=['user','item'],
		how='outer', suffixes=('', '_y'))
	train = train[train.rating_y.isnull()]
	test = test[['user','item','rating']]
	train = train[['user','item','rating']]

	# select 99 random movies per user that were never rated by that user
	all_items = dfc.item.unique()
	rated_items = (dfc.groupby("user")['item']
	    .apply(list)
	    .reset_index()
	    ).item.tolist()

	def sample_not_rated(item_list, rseed=1, n=99):
		np.random.seed=rseed
		return np.random.choice(np.setdiff1d(all_items, item_list), n)

	print("sampling not rated items...")
	start = time()
	non_rated_items = Parallel(n_jobs=4)(delayed(sample_not_rated)(ri) for ri in rated_items)
	end = time() - start
	print("sampling took {} min".format(round(end/60,2)))

	negative = pd.DataFrame({'negative':non_rated_items})
	negative[['item_n'+str(i) for i in range(99)]] =\
		pd.DataFrame(negative.negative.values.tolist(), index= negative.index)
	negative.drop('negative', axis=1, inplace=True)
	negative = negative.stack().reset_index()
	negative = negative.iloc[:, [0,2]]
	negative.columns = ['user','item']
	negative['rating'] = 0
	assert negative.shape[0] == len(non_rated_items)*99
	test_negative = (pd.concat([test,negative])
		.sort_values('user', ascending=True)
		.reset_index(drop=True)
		)
	# Ensuring that the 1st element every 100 is the rated item. This is
	# fundamental for testing
	test_negative.sort_values(['user', 'rating'], ascending=[True,False], inplace=True)
	assert np.all(test_negative.values[0::100][:,2] != 0)

	# Save
	np.savez(data_path/"neuralcf_split.npz", train=train.values, test=test.values,
		test_negative=test_negative.values, negatives=np.array(non_rated_items),
		n_users=n_users, n_items=n_items)

	# Save training as sparse matrix
	print("saving training set as sparse matrix...")
	train_mtx = array2mtx(train.values)
	save_npz(data_path/"neuralcf_train_sparse.npz", train_mtx)


if __name__ == '__main__':

	parser = argparse.ArgumentParser(description="Prepare the Amazon movies dataset.")
	parser.add_argument('--input_dir', type=str,
		default="/home/ubuntu/projects/RecoTour/datasets/Amazon",
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


	# Let's start preparing the data for a second training/testing approach,
	# identical to the one used here:
	# https://github.com/jrzaurin/RecoTour/tree/master/Amazon/neural_cf

	# rank of items bought
	print("preparing data for a NCF-like training/testing approach")
	df_c = copy(df)

	df_c['rank'] = df_c.groupby("user")["timestamp"].rank(ascending=True, method='dense')
	df_c.drop("timestamp", axis=1, inplace=True)

	# mapping user and item ids to integers
	user_mappings = {k:v for v,k in enumerate(df_c.user.unique())}
	item_mappings = {k:v for v,k in enumerate(df_c.item.unique())}
	df_c['user'] = df_c['user'].map(user_mappings)
	df_c['item'] = df_c['item'].map(item_mappings)
	df_c = df_c[['user','item','rank','rating']].astype(np.int64)

	neuralcf_split(df_c, DATA_PATH)
