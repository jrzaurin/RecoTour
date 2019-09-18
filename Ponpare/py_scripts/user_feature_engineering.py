import pandas as pd
import numpy as np
import argparse
import pickle
import os
import re

from functools import reduce
from scipy import stats
from collections import Counter


def top_values(row, top_n=2):
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


def user_features(inp_dir, out_dir):

	# User features: here we will consider ONLY users that were seeing during
	# training period. There is a caveat regarding to users that registered
	# long ago they had more time to interact. For those a possibility would
	# be recommending the most popular.

	# Interactions
	df_visits_train = pd.read_pickle(os.path.join(inp_dir, 'train', 'df_visits_train.p'))
	df_purchases_train = pd.read_pickle(os.path.join(inp_dir, 'train', 'df_purchases_train.p'))

	# User features
	df_users_train = pd.read_pickle(os.path.join(inp_dir, 'train', 'df_users_train.p'))

	# train coupons features
	df_coupon_train = pd.read_pickle(os.path.join(inp_dir, 'train', 'df_coupons_train_feat.p'))

	# dictionaty of mappings
	dict_of_mappings = pickle.load(open(os.path.join(inp_dir, 'dict_of_mappings.p'), 'rb') )

	# let's see if there are users in training that did nothing. No visits, no purchases
	active_users = list(
		set(
			list(df_visits_train.user_id_hash.unique()) +
			list(df_purchases_train.user_id_hash.unique())
			)
		)
	inactive_users = np.setdiff1d(list(df_users_train.user_id_hash.unique()), active_users)
	# 66 inactive_users

	# let's focus on active users. For convenience we use abbreviated names:

	# df_users_train_active (df_utr)
	df_utr = df_users_train[~df_users_train.user_id_hash.isin(inactive_users)]
	withdraw_users = df_utr[~df_utr.withdraw_date.isna()]['user_id_hash'].unique()
	# there are 922 users that have withdrawn, but we still keep them because
	# we still want to learn from their behaviour
	df_utr.drop(['reg_date','withdraw_date','days_to_present'], axis=1, inplace=True)

	# df_visits_train_active (df_vtr)
	df_vtr = df_visits_train[df_visits_train.user_id_hash.isin(df_utr.user_id_hash)]
	# df_purchases_train_active (df_ptr)
	df_ptr = df_purchases_train[df_purchases_train.user_id_hash.isin(df_utr.user_id_hash)]

	# there are 2117 coupons in the purchased table that are not in the visits
	# table. Maybe this is due to the fact that this coupons where purchased
	# without previous visits. A way to check this would be to see if all
	# purchased coupons in the visits table appear more than once
	visits_purchased = df_vtr[~df_vtr.purchaseid_hash.isna()]['view_coupon_id_hash'].unique()
	purchased_not_in_visits = df_ptr[~df_ptr.coupon_id_hash.isin(visits_purchased)]['coupon_id_hash'].unique()
	purchased_min_visits = (df_vtr[df_vtr.view_coupon_id_hash.isin(visits_purchased)]['view_coupon_id_hash']
		.value_counts()
		.min())
	# Effectively, all purchased coupons that are in the visit table must have
	# been visited at least once.

	# 1-. Let's start building user features based on "demographics" (_d)
	df_user_feat_d = df_utr.copy()

	df_user_feat_d['pref_name_cat'] = df_user_feat_d.pref_name.replace(dict_of_mappings['ken_name_cat'])
	new_pref_name_cat = df_user_feat_d.pref_name_cat.max() + 1
	df_user_feat_d['pref_name_cat'] = df_user_feat_d.pref_name_cat.fillna(new_pref_name_cat).astype('int')
	df_user_feat_d.drop('pref_name', axis=1, inplace=True)

	# given that we have added a category to pref_name, let's update the dict_of_mappings
	pref_name_dict = dict_of_mappings['ken_name_cat'].copy()
	pref_name_dict['NAN'] = int(new_pref_name_cat)
	dict_of_mappings['pref_name_cat'] = pref_name_dict

	dict_of_mappings['sex_id_cat'] = {'f':0, 'm':1}
	df_user_feat_d['sex_id_cat'] = df_user_feat_d.sex_id.replace(dict_of_mappings['sex_id_cat'])
	df_user_feat_d.drop('sex_id', axis=1, inplace=True)

	# 2-. Let's now build user features based on purchase behaviour (_p):
	df_ptr['day_of_week'] = df_ptr.i_date.dt.dayofweek

	agg_functions_p = {
		'purchaseid_hash': ['count'],
		'coupon_id_hash': ['nunique'],
		'item_count': ['sum'],
		'small_area_name': ['nunique'],
		'day_of_week': ['nunique']
		}
	df_user_feat_p = df_ptr.groupby("user_id_hash").agg(agg_functions_p)
	df_user_feat_p.columns =  ["_".join(pair) for pair in df_user_feat_p.columns]
	df_user_feat_p.reset_index(inplace=True)

	# median time difference between purchases
	time_diff_df_p = (df_ptr.groupby("user_id_hash")['days_to_present']
		.apply(list)
		.reset_index()
		)
	time_diff_df_p['median_time_diff'] = time_diff_df_p.days_to_present.apply(lambda x: time_diff(x))
	time_diff_df_p.drop('days_to_present', axis=1, inplace=True)

	# top 2 small area name of shop location and days of week for purchases
	small_area_df_p = (df_ptr.groupby("user_id_hash")['small_area_name']
		.apply(list)
		.reset_index()
		)
	small_area_df_p['top_small_areas'] = small_area_df_p.small_area_name.apply(lambda x: top_values(x))
	small_area_df_p['top1_small_area_name'] =  small_area_df_p.top_small_areas.apply(lambda x: x[0])
	small_area_df_p['top2_small_area_name'] =  small_area_df_p.top_small_areas.apply(lambda x: x[1])
	small_area_df_p.drop(['small_area_name', 'top_small_areas'], axis=1, inplace=True)
	for col in ['top1_small_area_name', 'top2_small_area_name']:
		small_area_df_p[col] = small_area_df_p[col].replace(dict_of_mappings['small_area_name_cat'])

	day_of_week_df_p = (df_ptr.groupby("user_id_hash")['day_of_week']
		.apply(list)
		.reset_index()
		)
	day_of_week_df_p['top_days_of_week'] = day_of_week_df_p.day_of_week.apply(lambda x: top_values(x))
	day_of_week_df_p['top1_dayofweek'] =  day_of_week_df_p.top_days_of_week.apply(lambda x: x[0])
	day_of_week_df_p['top2_dayofweek'] =  day_of_week_df_p.top_days_of_week.apply(lambda x: x[1])
	day_of_week_df_p.drop(['day_of_week', 'top_days_of_week'], axis=1, inplace=True)

	# merge all together
	df_l_p = [df_user_feat_p, time_diff_df_p, small_area_df_p, day_of_week_df_p]
	df_user_feat_p = reduce(lambda left,right: pd.merge(left,right,on=['user_id_hash']), df_l_p)

	# add "_cat" to categorical features
	cat_feat_p = ['top1_small_area_name', 'top2_small_area_name', 'top1_dayofweek', 'top2_dayofweek']
	cat_feat_name_p = [c+"_cat" for c in cat_feat_p]
	cat_feat_name_dict_p = dict(zip(cat_feat_p, cat_feat_name_p))
	df_user_feat_p.rename(index=str, columns=cat_feat_name_dict_p, inplace=True)

	# 3-. User features based on visit behaviour (_v):
	df_vtr['day_of_week'] = df_vtr.i_date.dt.dayofweek

	agg_functions_v = {
		'view_coupon_id_hash': ['count', 'nunique'],
		'session_id_hash': ['nunique'],
		'day_of_week': ['nunique']
		}
	df_user_feat_v = df_vtr.groupby("user_id_hash").agg(agg_functions_v)
	df_user_feat_v.columns =  ["_".join(pair) for pair in df_user_feat_v.columns]
	df_user_feat_v.reset_index(inplace=True)

	# min/max/median time difference between visits
	time_diff_df_v = (df_vtr.groupby("user_id_hash")['days_to_present']
		.apply(list)
		.reset_index()
		)
	time_diff_df_v['time_diff'] = time_diff_df_v.days_to_present.apply(lambda x: time_diff(x, all_metrics=True))
	time_diff_df_v['min_time_diff'] = time_diff_df_v.time_diff.apply(lambda x: x[0])
	time_diff_df_v['max_time_diff'] = time_diff_df_v.time_diff.apply(lambda x: x[1])
	time_diff_df_v['median_time_diff'] = time_diff_df_v.time_diff.apply(lambda x: x[2])
	time_diff_df_v.drop(['days_to_present','time_diff'], axis=1, inplace=True)

	# top 2 days of week for visits
	day_of_week_df_v = (df_vtr.groupby("user_id_hash")['day_of_week']
		.apply(list)
		.reset_index()
		)
	day_of_week_df_v['top_days_of_week'] = day_of_week_df_v.day_of_week.apply(lambda x: top_values(x))
	day_of_week_df_v['top1_dayofweek'] =  day_of_week_df_v.top_days_of_week.apply(lambda x: x[0])
	day_of_week_df_v['top2_dayofweek'] =  day_of_week_df_v.top_days_of_week.apply(lambda x: x[1])
	day_of_week_df_v.drop(['day_of_week', 'top_days_of_week'], axis=1, inplace=True)

	# merge all together
	df_l_v = [df_user_feat_v, time_diff_df_v, day_of_week_df_v]
	df_user_feat_v = reduce(lambda left,right: pd.merge(left,right,on=['user_id_hash']), df_l_v)

	cat_feat_v = ['top1_dayofweek', 'top2_dayofweek']
	cat_feat_name_v = [c+"_cat" for c in cat_feat_v]
	cat_feat_name_dict_v = dict(zip(cat_feat_v, cat_feat_name_v))
	df_user_feat_v.rename(index=str, columns=cat_feat_name_dict_v, inplace=True)

	# in the case of visits, let's add view to the beggining of all columns
	visits_cols = df_user_feat_v.columns.tolist()
	visits_cols = visits_cols[:3] + ['view_'+c for c in visits_cols[3:]]
	df_user_feat_v.columns = visits_cols

	# 4-. User features based on the coupons they bought (_c). Here
	# possibilities are nearly endless. Let's focus on price features
	# (catalogue price and price rate) and coupon category features:
	# capsule_text_cat and genre_name_cat.

	# remove purchases from the visits data because we will build features
	# separately for purchased and visits
	df_vtr_visits = df_vtr.copy()
	df_vtr_visits['activity_hash'] = df_vtr_visits['user_id_hash'] + "_" + df_vtr_visits['view_coupon_id_hash']
	purchases = df_vtr_visits[~df_vtr_visits.purchaseid_hash.isna()]['activity_hash'].unique()
	df_vtr_visits = df_vtr_visits[~df_vtr_visits.activity_hash.isin(purchases)][['user_id_hash','view_coupon_id_hash']]
	df_vtr_visits.columns = ['user_id_hash','coupon_id_hash']

	df_coupon_based_feat_l = []
	for name,df in [('purchase', df_ptr), ('visits', df_vtr_visits)]:

		list_user = ['user_id_hash', 'coupon_id_hash']
		list_coupons = ['coupon_id_hash', 'catalog_price', 'discount_price', 'catalog_price_cat',
			'discount_price_cat', 'capsule_text_cat', 'genre_name_cat']

		df_c = pd.merge(df[list_user], df_coupon_train[list_coupons], on='coupon_id_hash')
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
		if name is 'visits':
			new_colnames = old_colnames[:1] + ['view_'+c for c in old_colnames[1:]]
			fdf.columns = new_colnames

		df_coupon_based_feat_l.append(fdf)

	df_user_feat_c = reduce(lambda left,right: pd.merge(left,right,on=['user_id_hash'],how='outer'), df_coupon_based_feat_l)

	final_list_of_dfs = [df_user_feat_d, df_user_feat_c, df_user_feat_v, df_user_feat_p]
	df_user_feat = reduce(lambda left,right: pd.merge(left,right,on=['user_id_hash'],how='outer'), final_list_of_dfs)

	# there are 116 users in training that visit but never bought:
	# df_utr[(df_utr.user_id_hash.isin(df_vtr.user_id_hash)) & \
	# (~df_utr.user_id_hash.isin(df_ptr.user_id_hash))].shape

	# For them all columns in df_user_feat_p will be -1
	prefixes = ['top1_', 'top2_', 'top3_']
	for col in df_user_feat.columns[2:]:
		if col.endswith("_cat") and df_user_feat[col].isna().any():
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
				col_categories = np.sort(df_user_feat[col].unique())[:-1]
				col_dict = {int(k):int(v) for v,k in enumerate(col_categories)}
				new_col_cat =int(df_user_feat[col].max()+1)

			col_dict['NAN'] = new_col_cat
			dict_of_mappings[col] = col_dict
			df_user_feat[col] = df_user_feat[col].fillna(new_col_cat).astype('int')

		else:
			df_user_feat[col].fillna(-1, inplace=True)
	# There are 18 users that are in active_users but not in df_user_feat:
	# np.setdiff1d(active_users, df_users_train.user_id_hash.unique()).size

	# This is because these users are not in df_users_train

	# note that the "train" is redundant here. Unlike coupons, there will not be
	# user_feat_test in this excercise.
	df_user_feat.to_pickle(os.path.join(out_dir,"train","df_users_train_feat.p"))

	# save dictionary
	pickle.dump(dict_of_mappings, open(os.path.join(out_dir, 'dict_of_mappings.p'), 'wb') )


if __name__ == '__main__':

	parser = argparse.ArgumentParser(description="build the features for the coupons")

	parser.add_argument(
		"--root_data_dir",
		type=str, default="/home/ubuntu/projects/RecoTour/datasets/Ponpare/",)
	args = parser.parse_args()

	parser.add_argument("--input_dir",
		type=str, default=args.root_data_dir+"data_processed")
	parser.add_argument("--output_dir",
		type=str, default=args.root_data_dir+"data_processed")
	args = parser.parse_args()

	user_features(args.input_dir, args.output_dir)
