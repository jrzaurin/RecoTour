import pandas as pd
import numpy as np
import argparse
import os
import pickle
import pdb


from scipy import stats


def coupon_features(work_dir, is_validation):

	print('INFO: building coupong features')

	if is_validation:
		train_dir = os.path.join(work_dir, 'train')
		valid_dir = os.path.join(work_dir, 'valid')
		test_dir = os.path.join(work_dir, 'test')
		train_path = os.path.join(train_dir, 'df_coupons_train.p')
		valid_path = os.path.join(valid_dir, 'df_coupons_valid.p')
		test_path = os.path.join(test_dir, 'df_coupons_test.p')

		df_coupons_train = pd.read_pickle(train_path)
		df_coupons_valid = pd.read_pickle(valid_path)
		df_coupons_test  = pd.read_pickle(test_path)
		df_coupons_train['set_type'] = 2
		df_coupons_valid['set_type'] = 1
		df_coupons_test['set_type'] = 0

		df_coupons = pd.concat(
			[df_coupons_train,df_coupons_valid,df_coupons_test],
			axis=0,
			ignore_index=True)
	else:
		train_dir = os.path.join(work_dir, 'ftrain')
		test_dir = os.path.join(work_dir, 'test')
		train_path = os.path.join(train_dir, 'df_coupons_train.p')
		test_path = os.path.join(test_dir, 'df_coupons_test.p')

		df_coupons_train = pd.read_pickle(train_path)
		df_coupons_test  = pd.read_pickle(test_path)
		df_coupons_train['set_type'] = 2
		df_coupons_test['set_type'] = 0

		df_coupons = pd.concat(
			[df_coupons_train,df_coupons_test],
			axis=0,
			ignore_index=True)

	# columns with NaN
	has_nan = df_coupons.isnull().any(axis=0)
	has_nan = [df_coupons.columns[i] for i in np.where(has_nan)[0]]

	# All features with Nan are time related.
	# Usable_date_day has values of 0,1 and 2. I am going to replace NaN with
	# another value, 3, as if this was another category of coupons.
	for col in has_nan[3:]:
		df_coupons[col] = df_coupons[col].fillna(3).astype('int')

	# validperiod/validfrom/validend
	df_coupons, bins_method1 = fillna_method1(df_coupons, 'validperiod', q=4)
	df_coupons = fillna_method2(df_coupons, 'validperiod', 'capsule_text', isdayofweek=False, bins=bins_method1)
	df_coupons = fillna_method1(df_coupons, 'validfrom', isdayofweek=True)
	df_coupons = fillna_method2(df_coupons, 'validfrom', 'capsule_text', isdayofweek=True)
	df_coupons = fillna_method1(df_coupons, 'validend', isdayofweek=True)
	df_coupons = fillna_method2(df_coupons, 'validend', 'capsule_text', isdayofweek=True)

	dict_of_mappings = {}
	dict_of_mappings['validperiod_cat'] = bins_method1

	# dispperiod/dispfrom/dispend
	df_coupons['dispfrom_cat'] = df_coupons.dispfrom.dt.dayofweek
	df_coupons['dispend_cat'] = df_coupons.dispend.dt.dayofweek
	df_coupons['dispperiod_cat'], dispperiod_bins = pd.qcut(df_coupons.dispperiod, q=4, labels=[0,1,2,3], retbins=True)
	dict_of_mappings['dispperiod_cat'] = dispperiod_bins

	# price related features
	df_coupons['price_rate_cat'], price_rate_bins = pd.qcut(df_coupons['price_rate'], q=3, labels=[0,1,2], retbins=True)
	df_coupons['catalog_price_cat'], catalog_price_bins = pd.qcut(df_coupons['catalog_price'], q=3, labels=[0,1,2], retbins=True)
	df_coupons['discount_price_cat'], discount_price_bins = pd.qcut(df_coupons['discount_price'], q=3, labels=[0,1,2], retbins=True)
	dict_of_mappings['price_rate_cat'] = price_rate_bins
	dict_of_mappings['catalog_price_cat'] = catalog_price_bins
	dict_of_mappings['discount_price_cat'] = discount_price_bins

	# LabelEncode some additional features. I will do it manually since I want
	# to keep the mappings
	le_cols = ['capsule_text', 'genre_name', 'large_area_name', 'ken_name', 'small_area_name']
	for col in le_cols:
		dict_of_mappings[col+"_cat"] = {}
		values = list(df_coupons[col].unique())
		labels = list(np.arange(len(values)))
		dict_of_mappings[col+"_cat"] = dict(zip(values,labels))
		df_coupons[col+'_cat'] = df_coupons[col].replace(dict_of_mappings[col+"_cat"])

	# Drop redundant columns
	drop_cols = le_cols + ['dispfrom', 'dispend', 'validfrom', 'validend', 'days_to_present']
	df_coupons.drop(drop_cols, axis=1, inplace=True)

	# for convenience, let's add the suffix "_cat" to usable_date columns
	usable_date_cols_dict = {c:c+"_cat" for c in df_coupons.columns if 'usable' in c}
	df_coupons.rename(index=str, columns=usable_date_cols_dict, inplace=True)

	# split back to the originals
	df_coupons_train = (df_coupons[df_coupons.set_type == 2]
		.drop('set_type', axis=1)
		.reset_index(drop=True))
	df_coupons_valid = (df_coupons[df_coupons.set_type == 1]
		.drop('set_type', axis=1)
		.reset_index(drop=True))
	df_coupons_test = (df_coupons[df_coupons.set_type == 0]
		.drop('set_type', axis=1)
		.reset_index(drop=True))

	# save files
	if is_validation:
		df_coupons_train.to_pickle(os.path.join(train_dir,"df_coupons_train_feat.p"))
		df_coupons_valid.to_pickle(os.path.join(valid_dir,"df_coupons_valid_feat.p"))
		df_coupons_test.to_pickle(os.path.join(test_dir,"df_coupons_test_feat.p"))
	else:
		df_coupons_train.to_pickle(os.path.join(train_dir,"df_coupons_train_feat.p"))
		df_coupons_test.to_pickle(os.path.join(test_dir,"df_coupons_test_feat.p"))

	# save dictionary of mappings
	pickle.dump(dict_of_mappings, open(os.path.join(work_dir, 'dict_of_mappings.p'), 'wb') )


def fillna_method1(df, col, isdayofweek=False, q=None):
	"""
	fill NaN simply considering them as a new category
	"""
	newcolname = col + "_method1_cat"

	if isdayofweek:
		df[newcolname] = df[col].dt.dayofweek
		df[newcolname] = df[newcolname].fillna(7).astype('int')
		return df
	else:
		df[newcolname], bins_method1 = pd.qcut(df[col], q=q, labels=np.arange(q), retbins=True)
		df[newcolname].cat.add_categories([q], inplace=True)
		df[newcolname].fillna(q, inplace=True)
		return df, bins_method1


def fillna_method2(df, col, fillby, nanlimit=50, seed=1981, isdayofweek=False, bins=None):
	"""
	fill NaN replacing values per category
	"""

	newcolname = col + "_method2_cat"

	if isdayofweek:
		colmode = df[col].dt.dayofweek.mode().values[0]
		df[newcolname] = df[col].dt.dayofweek
		df = fill_loop(df, newcolname, fillby, colmode, nanlimit, seed, isdayofweek)
		df[newcolname] = df[newcolname].astype('int')
	else:
		colmode = df[col].mode().values[0]
		df = fill_loop(df, col, fillby, colmode, nanlimit, seed, isdayofweek)
		df[newcolname] = pd.cut(df[col], bins=bins, labels=np.arange(len(bins)-1), include_lowest=True)
		df[col] = df[col].astype('int')

	return df


def fill_loop(df, col, fillby, colmode, nanlimit, seed, isdayofweek):
	"""
	conditions to be used to fill NaN when using method 2.
	We fill per category.

	cond1: there are more NaN than nanlimit and at least one non NaN value
		replace at random
	cond2: there are less NaN than nanlimit and at least one non NaN value
		replace with the mode if isdayofweek, else with the median
	cond3: there are non non_nan_values for that particular category
		replace with colmode
	"""
	fillby_vals = list(df[df[col].isna()][fillby].value_counts().index)

	for fb in fillby_vals:
		non_nan_values = df[(df[fillby] == fb) & (~df[col].isna())][col].values
		nan_idx = list(df[(df[fillby] == fb) & (df[col].isna())].index)
		if (len(nan_idx)>nanlimit) & (non_nan_values.size>0):
			np.random.seed(seed)
			replace_vals = np.random.choice(non_nan_values, len(nan_idx))
			df.loc[nan_idx, col] = replace_vals
		elif (len(nan_idx)<=nanlimit) & (non_nan_values.size>0):
			df.loc[nan_idx, col] = \
			stats.mode(non_nan_values).mode[0] if isdayofweek else np.median(non_nan_values)
		elif non_nan_values.size==0:
			df.loc[nan_idx, col] = colmode

	return df


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

	coupon_features(
		args.work_dir,
		args.is_validation
		)