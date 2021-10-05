import pandas as pd
import numpy as np
import argparse
import pickle
import os
import re

from functools import reduce
from collections import Counter


def user_features(work_dir, is_validation):

	print('INFO: building user features')

	if is_validation:
		train_dir = os.path.join(work_dir, 'train')
	else:
		train_dir = os.path.join(work_dir, 'ftrain')

	# Load interactions, user features, train coupons features and dictionary
	# of mappings
	df_visits_train = pd.read_pickle(os.path.join(train_dir, 'df_visits_train.p'))
	df_purchases_train = pd.read_pickle(os.path.join(train_dir, 'df_purchases_train.p'))
	df_users_train = pd.read_pickle(os.path.join(train_dir, 'df_users_train.p'))
	df_coupon_train = pd.read_pickle(os.path.join(train_dir, 'df_coupons_train_feat.p'))
	dict_of_mappings = pickle.load(open(os.path.join(work_dir, 'dict_of_mappings.p'), 'rb') )

	active_users = list(
		set(
			list(df_visits_train.user_id_hash.unique()) +
			list(df_purchases_train.user_id_hash.unique())
			)
		)
	inactive_users = np.setdiff1d(list(df_users_train.user_id_hash.unique()), active_users)

	# df_users_train_active     (df_utr)
	# df_visits_train_active    (df_vtr)
	# df_purchases_train_active (df_ptr)
	df_utr = df_users_train[~df_users_train.user_id_hash.isin(inactive_users)]
	df_utr.drop(['reg_date','withdraw_date','days_to_present'], axis=1, inplace=True)
	df_vtr = df_visits_train[df_visits_train.user_id_hash.isin(df_utr.user_id_hash)]
	df_ptr = df_purchases_train[df_purchases_train.user_id_hash.isin(df_utr.user_id_hash)]

	# 1. Features based on "demographics" (_d)
	dict_of_mappings, df_user_feat_d = demographic_features(df_utr , dict_of_mappings)

	# 2. Features based on purchase behaviour (_p):
	df_user_feat_p = purchase_behaviour_features(df_ptr , dict_of_mappings)

	# 3. User features based on visit behaviour (_v):
	df_user_feat_v = visits_behaviour_features(df_vtr)

	# 4. User features based on "general behaviour" (type, price, etc)
	df_user_feat_g_p = general_behaviour_features(df_ptr, df_coupon_train, action='purchases')
	df_user_feat_g_v = general_behaviour_features(df_vtr, df_coupon_train, action='visits')
	df_user_feat_g = pd.merge(df_user_feat_g_p, df_user_feat_g_v, on='user_id_hash', how='outer')

	# 5. merge all dataframes
	final_list_of_dfs = [df_user_feat_d, df_user_feat_g, df_user_feat_v, df_user_feat_p]
	df_user_feat = reduce(lambda left,right: pd.merge(left,right,on=['user_id_hash'],how='outer'), final_list_of_dfs)

	# There are 116 users in training that visit but never bought:
	# df_utr[(df_utr.user_id_hash.isin(df_vtr.user_id_hash)) & \
	# (~df_utr.user_id_hash.isin(df_ptr.user_id_hash))].shape
	# For them all columns in df_user_feat_p will be -1
	dict_of_mappings, df_user_feat = fillna_with_minus_one(df_user_feat, dict_of_mappings)

	# Save files
	df_user_feat.to_pickle(os.path.join(train_dir,"df_users_train_feat.p"))
	pickle.dump(dict_of_mappings, open(os.path.join(work_dir, 'dict_of_mappings.p'), 'wb') )


def demographic_features(df_inp, dict_of_mappings):

	df = df_inp.copy()

	df['pref_name_cat'] = df.pref_name.replace(dict_of_mappings['ken_name_cat'])
	new_pref_name_cat = df.pref_name_cat.max() + 1
	df['pref_name_cat'] = df.pref_name_cat.fillna(new_pref_name_cat).astype('int')
	df.drop('pref_name', axis=1, inplace=True)

	# update the dict_of_mappings
	pref_name_dict = dict_of_mappings['ken_name_cat'].copy()
	pref_name_dict['NAN'] = int(new_pref_name_cat)
	dict_of_mappings['pref_name_cat'] = pref_name_dict

	dict_of_mappings['sex_id_cat'] = {'f':0, 'm':1}
	df['sex_id_cat'] = df.sex_id.replace(dict_of_mappings['sex_id_cat'])
	df.drop('sex_id', axis=1, inplace=True)

	return dict_of_mappings, df


def purchase_behaviour_features(df_inp, mappings):

	df_pbf = df_inp.copy()
	df_pbf['day_of_week'] = df_pbf.i_date.dt.dayofweek

	agg_functions_p = {
		'purchaseid_hash': ['count'],
		'coupon_id_hash': ['nunique'],
		'item_count': ['sum'],
		'small_area_name': ['nunique'],
		'day_of_week': ['nunique']
		}
	df = df_pbf.groupby("user_id_hash").agg(agg_functions_p)
	df.columns =  ["_".join(pair) for pair in df.columns]
	df.reset_index(inplace=True)

	# median time difference between purchases
	time_diff_df = (df_pbf.groupby("user_id_hash")['days_to_present']
		.apply(list)
		.reset_index()
		)
	time_diff_df['median_time_diff'] = time_diff_df.days_to_present.apply(lambda x: time_diff(x))
	time_diff_df.drop('days_to_present', axis=1, inplace=True)

	# top 2 small area name
	small_area_df = top_values_df(df_pbf, 'small_area_name', mappings)

	# top 2 day of the week
	day_of_week_df = top_values_df(df_pbf, 'day_of_week')

	# merge all dataframes
	df_l = [df, time_diff_df, small_area_df, day_of_week_df]
	df = reduce(lambda left,right: pd.merge(left,right,on=['user_id_hash']), df_l)

	# add "_cat" to categorical features
	cat_feat_p = ['top1_small_area_name', 'top2_small_area_name', 'top1_day_of_week', 'top2_day_of_week']
	cat_feat_name_p = [c+"_cat" for c in cat_feat_p]
	cat_feat_name_dict_p = dict(zip(cat_feat_p, cat_feat_name_p))
	df.rename(index=str, columns=cat_feat_name_dict_p, inplace=True)

	return df


def visits_behaviour_features(df_inp):

	df_vbf = df_inp.copy()
	df_vbf['day_of_week'] = df_vbf.i_date.dt.dayofweek

	agg_functions_v = {
		'view_coupon_id_hash': ['count', 'nunique'],
		'session_id_hash': ['nunique'],
		'day_of_week': ['nunique']
		}
	df = df_vbf.groupby("user_id_hash").agg(agg_functions_v)
	df.columns =  ["_".join(pair) for pair in df.columns]
	df.reset_index(inplace=True)

	# min/max/median time difference between visits
	time_diff_df = (df_vbf.groupby("user_id_hash")['days_to_present']
		.apply(list)
		.reset_index()
		)
	time_diff_df['time_diff'] = time_diff_df.days_to_present.apply(lambda x: time_diff(x, all_metrics=True))
	time_diff_df['min_time_diff'] = time_diff_df.time_diff.apply(lambda x: x[0])
	time_diff_df['max_time_diff'] = time_diff_df.time_diff.apply(lambda x: x[1])
	time_diff_df['median_time_diff'] = time_diff_df.time_diff.apply(lambda x: x[2])
	time_diff_df.drop(['days_to_present','time_diff'], axis=1, inplace=True)

	# top 2 days of week
	day_of_week_df = top_values_df(df_vbf, 'day_of_week')

	# merge all df togetehr
	df_l = [df, time_diff_df, day_of_week_df]
	df = reduce(lambda left,right: pd.merge(left,right,on=['user_id_hash']), df_l)

	cat_feat_v = ['top1_day_of_week', 'top2_day_of_week']
	cat_feat_name_v = [c+"_cat" for c in cat_feat_v]
	cat_feat_name_dict_v = dict(zip(cat_feat_v, cat_feat_name_v))
	df.rename(index=str, columns=cat_feat_name_dict_v, inplace=True)

	# in the case of visits, let's add view to the beggining of all columns
	visits_cols = df.columns.tolist()
	visits_cols = visits_cols[:3] + ['view_'+c for c in visits_cols[3:]]
	df.columns = visits_cols

	return df


def general_behaviour_features(df_inp, df_coupon_features, action):

	if action is 'visits':
		df = df_inp.copy()
		df['activity_hash'] = df['user_id_hash'] + "_" + df['view_coupon_id_hash']
		purchases = df[~df.purchaseid_hash.isna()]['activity_hash'].unique()
		df = df[~df.activity_hash.isin(purchases)][['user_id_hash','view_coupon_id_hash']]
		df.columns = ['user_id_hash','coupon_id_hash']
	else:
		df = df_inp.copy()

	list_user = ['user_id_hash', 'coupon_id_hash']
	list_coupons = ['coupon_id_hash', 'catalog_price', 'discount_price', 'catalog_price_cat',
		'discount_price_cat', 'capsule_text_cat', 'genre_name_cat']

	df_c = pd.merge(df[list_user], df_coupon_features[list_coupons], on='coupon_id_hash')
	agg_functions_p_c = {
		'catalog_price': ['mean', 'median', 'min', 'max'],
		'discount_price': ['mean', 'median', 'min', 'max'],
		}
	df_price_num = df_c.groupby('user_id_hash').agg(agg_functions_p_c)
	df_price_num.columns =  ["_".join(pair) for pair in df_price_num.columns]
	df_price_num.reset_index(inplace=True)

	tmp_df_l_1 = []
	for col in ['catalog_price_cat', 'discount_price_cat']:
		tmp_df = df_c.pivot_table(values='coupon_id_hash', index='user_id_hash',
			columns=col, aggfunc= lambda x: len(x.unique()))
		colname = col.split("_cat")[0]
		colnames = ["_".join([colname,str(cat)]) for cat in tmp_df.columns.categories]
		tmp_df.columns = colnames
		tmp_df.reset_index(inplace=True)
		tmp_df.fillna(0, inplace=True)
		tmp_df_l_1.append(tmp_df)
	df_price_cat = reduce(lambda left,right: pd.merge(left,right,on=['user_id_hash']), tmp_df_l_1)

	tmp_df_l_2 = []
	top_n = 3
	for col in ['capsule_text_cat', 'genre_name_cat']:
		tmp_df = df_c.groupby('user_id_hash')[col].apply(list).reset_index()
		root = col.split("_cat")[0]
		colname = "top_" + root
		colnames = ["top"+str(i)+"_"+root for i in range(1,top_n+1)]
		tmp_df[colname] = tmp_df[col].apply(lambda x: top_values(x, top_n=top_n))
		for i,cn in enumerate(colnames):
			tmp_df[cn] = tmp_df[colname].apply(lambda x: x[i])
		tmp_df.drop([col, colname], axis=1, inplace=True)
		tmp_df_l_2.append(tmp_df)
	df_type_cat = reduce(lambda left,right: pd.merge(left,right,on=['user_id_hash']), tmp_df_l_2)
	tmp_colnames = df_type_cat.columns.tolist()
	cat_colnames = tmp_colnames[:1]+[c+"_cat" for c in tmp_colnames[1:]]
	df_type_cat.columns = cat_colnames

	df_l_c = [df_price_num, df_price_cat, df_type_cat]
	fdf = reduce(lambda left,right: pd.merge(left,right,on=['user_id_hash']), df_l_c)
	old_colnames = fdf.columns.tolist()

	if action is 'visits':
		new_colnames = old_colnames[:1] + ['view_'+c for c in old_colnames[1:]]
		fdf.columns = new_colnames

	return fdf


def fillna_with_minus_one(df_inp, dict_of_mappings):

	prefixes = ['top1_', 'top2_', 'top3_']
	for col in df_inp.columns[2:]:
		if col.endswith("_cat") and df_inp[col].isna().any():
			# We are going to treat NaN as a new category for these cols, so
			# we need to update the dictionary of mappings. They are mostly
			# related to capsule_text_cat and genre_name_cat with the prefix
			# top_n:
			for prefix in prefixes:
				if prefix in col:
					start = re.search(prefix, col).end()
			root_name = col[start:]

			# if the column is derived from root_name, the correponding
			# dictionary would be the same as that of root_name plus an extra
			# category for 'NAN'
			if root_name in dict_of_mappings.keys():
				col_dict = dict_of_mappings[root_name].copy()
				new_col_cat = len(col_dict)
			else:
				col_categories = np.sort(df_inp[col].unique())[:-1]
				col_dict = {int(k):int(v) for v,k in enumerate(col_categories)}
				new_col_cat =int(df_inp[col].max()+1)

			col_dict['NAN'] = new_col_cat
			dict_of_mappings[col] = col_dict
			df_inp[col] = df_inp[col].fillna(new_col_cat).astype('int')

		else:
			df_inp[col].fillna(-1, inplace=True)

	return dict_of_mappings, df_inp


def top_values_df(df, col, mappings=None, top_n=2):
	"""
	Returns a dataframe with columns are the top "tokens" the user interacted
	with. For example, topN coupon category
	"""
	newcolname = 'top_' + col
	topcolnames = ['top'+ str(i+1) + '_' + col for i in range(top_n)]

	groupbydf = (df.groupby("user_id_hash")[col]
		.apply(list)
		.reset_index()
		)
	groupbydf[newcolname] = groupbydf[col].apply(lambda x: top_values(x, top_n=top_n))

	for i,c in enumerate(topcolnames):
		groupbydf[c] =  groupbydf[newcolname].apply(lambda x: x[i])
	groupbydf.drop([col, newcolname], axis=1, inplace=True)

	if mappings:
		for c in topcolnames:
			groupbydf[c] = groupbydf[c].replace(mappings[col+'_cat'])

	return groupbydf


def top_values(row, top_n=2):
	"""
	Helper function that returns a list with the  top "tokens" the user
	interacted with (on a row basis)
	"""
	counts = [c[0] for c in Counter(row).most_common()]
	row_len = len(set(row))
	if row_len < top_n:
		top_vals = counts + [counts[-1]]*(top_n - row_len)
	else:
		top_vals = counts[:top_n]
	return top_vals


def time_diff(row, all_metrics=False):
	if len(row) == 1:
		min_diff = 0
		max_diff = 0
		median_diff = 0
	else:
		row = sorted(row, reverse=True)
		diff = [t - s for t, s in zip(row, row[1:])]
		min_diff = np.min(diff)
		max_diff = np.max(diff)
		median_diff = np.median(diff)
	if all_metrics:
		return [min_diff, max_diff, median_diff]
	else:
		return median_diff


if __name__ == '__main__':

	parser = argparse.ArgumentParser(description="build the features for the coupons")

	parser.add_argument(
		"--root_data_dir",
		type=str, default="/home/ubuntu/projects/RecoTour/datasets/Ponpare/",)
	args = parser.parse_args()

	parser.add_argument("--work_dir",
		type=str, default=args.root_data_dir+"data_processed")
	parser.add_argument("--is_validation",
		action='store_false')

	args = parser.parse_args()

	user_features(
		args.work_dir,
		args.is_validation
		)