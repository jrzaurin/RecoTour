import pandas as pd
import numpy as np
import argparse
import os
import pickle

from scipy import stats


def coupon_features(inp_dir, out_dir):

	# We will assume that we know the coupons that will go live beforehand and
	# that we have the time to compute the features using the whole dataset of
	# coupons

	# Coupon features
	df_coupons_train = pd.read_pickle(os.path.join(inp_dir, 'train', 'df_coupons_train.p'))
	df_coupons_train['set_type'] = 2
	df_coupons_valid = pd.read_pickle(os.path.join(inp_dir, 'valid', 'df_coupons_valid.p'))
	df_coupons_valid['set_type'] = 1
	df_coupons_test  = pd.read_pickle(os.path.join(inp_dir, 'test', 'df_coupons_test.p'))
	df_coupons_test['set_type'] = 0
	df_coupons = pd.concat(
		[df_coupons_train,df_coupons_valid,df_coupons_test],
		axis=0,
		ignore_index=True)

	# columns with NaN
	has_nan = df_coupons.isnull().any(axis=0)
	has_nan = [df_coupons.columns[i] for i in np.where(has_nan)[0]]

	# All features with Nan are time related features have nan.
	# usable_date_day has values of 0,1 and 2, so I am going to replace NaN
	# with another value: 3, as if this was another category of coupons.
	for col in has_nan[3:]:
		df_coupons[col] = df_coupons[col].fillna(3).astype('int')

	# 6147 of the validperiod entries are NaN. 5821 are "Delivery service". In
	# the case of "Correspondence course" all observations have validperiod
	# NaN and "Other" and "Leisure" only have one entry with NaN. There is an
	# option that those coupons with no validperiod last "forever". Therefore,
	# to start with, we will create a categorical feature grouping and
	# defining a new category coupons with no valid period

	# Method 1: NaN as another category. Maybe we want to consider coupons
	# with a validperiod of 0 as a class. For now, we will simply use quatiles
	df_coupons['validperiod_method1_cat'], validperiod_bins_method1 = pd.qcut(df_coupons['validperiod'], q=4, labels=[0,1,2,3], retbins=True)
	df_coupons.validperiod_method1_cat.cat.add_categories([4], inplace=True)
	df_coupons['validperiod_method1_cat'].fillna(4, inplace=True)

	# Method 2: replace NaN first and then into categorical
	validperiod_mode = df_coupons.validperiod.mode().values[0]
	caps_vals = list(df_coupons[df_coupons.validperiod.isna()]['capsule_text'].value_counts().index)
	for ct in caps_vals:
		non_nan_values = df_coupons[(df_coupons.capsule_text == ct) &
			(~df_coupons.validperiod.isna())]['validperiod'].values
		nan_idx = list(df_coupons[(df_coupons.capsule_text == ct) &
			(df_coupons.validperiod.isna())].index)
		if (len(nan_idx)>50) & (non_nan_values.size>0):
			replace_vals = np.random.choice(non_nan_values, len(nan_idx))
			df_coupons.loc[nan_idx, 'validperiod'] = replace_vals
		elif (len(nan_idx)<=50) & (non_nan_values.size>0):
			median = np.median(non_nan_values)
			df_coupons.loc[nan_idx, 'validperiod'] = median
		elif non_nan_values.size==0:
			df_coupons.loc[nan_idx, 'validperiod'] = validperiod_mode
	df_coupons['validperiod_method2_cat'] = pd.cut(df_coupons['validperiod'],
		bins=validperiod_bins_method1, labels=[0,1,2,3], include_lowest=True)
	df_coupons['validperiod'] = df_coupons['validperiod'].astype('int')

	# let's save the mappings
	dict_of_mappings = {}
	dict_of_mappings['validperiod_cat'] = validperiod_bins_method1

	# Let's now take care of validfrom and validend
	valid_cols = ['validfrom', 'validend']
	for col in valid_cols:

		valid_mode = df_coupons[col].dt.dayofweek.mode().values[0]

		# Method 1: NaN as another category
		new_colname_1 = col+"_method1_cat"
		df_coupons[new_colname_1] = df_coupons[col].dt.dayofweek
		df_coupons[new_colname_1] = df_coupons[new_colname_1].fillna(7).astype('int')

		# Method 2: replace per category
		new_colname_2 = col+"_method2_cat"
		df_coupons[new_colname_2] = df_coupons[col].dt.dayofweek
		caps_vals = list(df_coupons[df_coupons[new_colname_2].isna()]['capsule_text'].value_counts().index)
		for ct in caps_vals:
			non_nan_values = df_coupons[(df_coupons.capsule_text == ct) &
				(~df_coupons[new_colname_2].isna())][new_colname_2].values
			nan_idx = list(df_coupons[(df_coupons.capsule_text == ct) &
				(df_coupons[new_colname_2].isna())].index)
			if (len(nan_idx)>50) & (non_nan_values.size>0):
				replace_vals = np.random.choice(non_nan_values, len(nan_idx))
				df_coupons.loc[nan_idx, new_colname_2] = replace_vals
			elif (len(nan_idx)<=50) & (non_nan_values.size>0):
				mode = stats.mode(non_nan_values).mode[0]
				df_coupons.loc[nan_idx, new_colname_2] = mode
			elif non_nan_values.size==0:
				df_coupons.loc[nan_idx, new_colname_2] = valid_mode
		df_coupons[new_colname_2] = df_coupons[new_colname_2].astype('int')

	# let's do dispfrom/dispend and dispperiod
	df_coupons['dispfrom_cat'] = df_coupons.dispfrom.dt.dayofweek
	df_coupons['dispend_cat'] = df_coupons.dispend.dt.dayofweek
	# and also add dispperiod as categorical
	df_coupons['dispperiod_cat'], dispperiod_bins = pd.qcut(df_coupons.dispperiod, q=4, labels=[0,1,2,3], retbins=True)
	dict_of_mappings['dispperiod_cat'] = dispperiod_bins

	# price related features
	df_coupons['price_rate_cat'], price_rate_bins = pd.qcut(df_coupons['price_rate'], q=3, labels=[0,1,2], retbins=True)
	df_coupons['catalog_price_cat'], catalog_price_bins = pd.qcut(df_coupons['catalog_price'], q=3, labels=[0,1,2], retbins=True)
	df_coupons['discount_price_cat'], discount_price_bins = pd.qcut(df_coupons['discount_price'], q=3, labels=[0,1,2], retbins=True)
	dict_of_mappings['price_rate_cat'] = price_rate_bins
	dict_of_mappings['catalog_price_cat'] = catalog_price_bins
	dict_of_mappings['discount_price_cat'] = discount_price_bins

	# Finally let's LabelEncode some additional features. I will do it
	# manually since I want to keep the mappings
	le_cols = ['capsule_text', 'genre_name', 'large_area_name', 'ken_name', 'small_area_name']
	for col in le_cols:
		dict_of_mappings[col+"_cat"] = {}
		values = list(df_coupons[col].unique())
		labels = list(np.arange(len(values)))
		dict_of_mappings[col+"_cat"] = dict(zip(values,labels))
		df_coupons[col+'_cat'] = df_coupons[col].replace(dict_of_mappings[col+"_cat"])


	drop_cols = le_cols + ['dispfrom', 'dispend', 'validfrom', 'validend', 'days_to_present']
	df_coupons.drop(drop_cols, axis=1, inplace=True)

	# for convenience later, let's add the suffix "cat" to usable_date columns
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
	df_coupons_train.to_pickle(os.path.join(out_dir,"train","df_coupons_train_feat.p"))
	df_coupons_valid.to_pickle(os.path.join(out_dir,"valid","df_coupons_valid_feat.p"))
	df_coupons_test.to_pickle(os.path.join(out_dir,"test","df_coupons_test_feat.p"))

	# save dictionary
	pickle.dump(dict_of_mappings, open(os.path.join(out_dir, 'dict_of_mappings.p'), 'wb') )

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description="build the features for the coupons")

	parser.add_argument("--input_dir", type=str, default="../datasets/Ponpare/data_processed")
	parser.add_argument("--output_dir", type=str, default="../datasets/Ponpare/data_processed")
	args = parser.parse_args()

	coupon_features(args.input_dir, args.output_dir)
