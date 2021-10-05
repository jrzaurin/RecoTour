'''
Direct copy of:
final_recommendations/recolearn/dataprep_utils/set_experiment.py

renamed to split_dataset.py for consistency with the notebooks, since the code
in final_recommendations is meant to be more of a production oriented code

to run simply:
python split_dataset.py
'''

import numpy as np
import pandas as pd
import argparse
import os


def split_data(input_dir, output_dir, users_list, coupon_list,
	purchase_log, viewing_log, test_period, is_validation):

	"""
	highly customized function to split the dataset into train/val/test

	Params (only describing those that are not trivial to understand)
	---------
	test_period: integer
		time range (days) to be used to split the data
	"""

	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	# the master list of users in the dataset
	df_users = pd.read_csv(os.path.join(input_dir,users_list))
	df_users['reg_date'] =  pd.to_datetime(df_users.reg_date, infer_datetime_format=True)

	# master list of coupons
	df_coupons = pd.read_csv(os.path.join(input_dir,coupon_list))
	df_coupons['dispfrom'] = pd.to_datetime(df_coupons.dispfrom, infer_datetime_format=True)
	df_coupons['dispend'] = pd.to_datetime(df_coupons.dispend, infer_datetime_format=True)
	df_coupons['validfrom'] = pd.to_datetime(df_coupons.validfrom, infer_datetime_format=True)
	df_coupons['validend'] = pd.to_datetime(df_coupons.validend, infer_datetime_format=True)

	# the purchase log of users buying coupons during training
	df_purchases = pd.read_csv(os.path.join(input_dir,purchase_log))
	df_purchases['i_date'] = pd.to_datetime(df_purchases.i_date, infer_datetime_format=True)

	# the viewing log of users browsing coupons during training
	df_visits = pd.read_csv(os.path.join(input_dir,viewing_log))
	df_visits['i_date'] = pd.to_datetime(df_visits.i_date, infer_datetime_format=True)

	# Find most recent date
	df_interactions = [df_visits, df_purchases]
	most_recent = []
	for df in df_interactions:
		for col in df.columns:
			if col == 'i_date':
				most_recent.append(df[col].max())
	present = np.max(most_recent)

	save_dset(df_visits, "df_visits", "i_date", present, test_period, output_dir, is_validation)
	save_dset(df_purchases, "df_purchases", "i_date", present, test_period, output_dir, is_validation)
	save_dset(df_users, "df_users", "reg_date", present, test_period, output_dir, is_validation)
	save_dset(df_coupons, "df_coupons", "dispfrom", present, test_period, output_dir, is_validation)


def save_dset(df, df_name, date_column, present, test_period, output_dir, is_validation):
	"""
	Simple helper to structure and save the datasets depending on the set up,
	whether is using a validation dataset or not.
	"""

	print('INFO: splitting {}'.format(df_name.split('_')[1]))
	# set names depending the experiment we are running
	if is_validation:
		train_dir  = os.path.join(output_dir,"train")
		test_dir   = os.path.join(output_dir,"test")
		valid_dir  = os.path.join(output_dir,"valid")

		train_path = os.path.join(train_dir,df_name+'_train.p')
		test_path  = os.path.join(test_dir, df_name+'_test.p')
		valid_path = os.path.join(valid_dir,df_name+'_valid.p')
	else:
		train_dir = os.path.join(output_dir,"ftrain")
		test_dir  = os.path.join(output_dir,"test")

		train_path = os.path.join(output_dir,train_dir,df_name+'_train.p')
		test_path  = os.path.join(output_dir,test_dir, df_name+'_test.p')

	# add train/validation/test flag
	fdf = flag_dset(df, date_column, present, test_period, is_validation)

	# train
	if not os.path.exists(train_dir): os.makedirs(train_dir)
	tmp_train = (fdf[fdf['dset'] == 2]
		.drop('dset', axis=1)
		.reset_index(drop=True))
	tmp_train.to_pickle(train_path)

	# test
	if not os.path.exists(test_dir): os.makedirs(test_dir)
	tmp_test  = (fdf[fdf['dset'] == 0]
		.drop('dset', axis=1)
		.reset_index(drop=True))
	tmp_test.to_pickle(test_path)

	# validation
	if is_validation:
		if not os.path.exists(valid_dir): os.makedirs(valid_dir)
		tmp_valid = (fdf[fdf['dset'] == 1]
			.drop('dset', axis=1)
			.reset_index(drop=True))
		tmp_valid.to_pickle(valid_path)


def flag_dset(df, date_column, present, test_period, is_validation):
	"""
	helper to flag data (0,1,2) depending on whether the dataset is
	(test,val,train)
	"""
	df = days_to_present_col(df, present, date_column)

	if is_validation:
		df['dset'] = df.days_to_present.apply(
			lambda x: 0 if x<=test_period-1 else 1
			if ((x>test_period-1) and (x<=(test_period*2)-1)) else 2)
	else:
		df['dset'] = df.days_to_present.apply(
			lambda x: 0 if x<=test_period-1 else 2)

	return df


def days_to_present_col(df, present, date_column):

	tmp_df = pd.DataFrame({'present': [present]*df.shape[0]})
	df['days_to_present'] = (tmp_df['present'] - df[date_column])
	df['days_to_present'] = df.days_to_present.dt.days

	return df


if __name__ == '__main__':

	parser = argparse.ArgumentParser(description="create train/valid/test sets based on a given testing period")

	parser.add_argument(
		"--root_data_dir",
		type=str, default="/home/ubuntu/projects/RecoTour/datasets/Ponpare/",)
	args = parser.parse_args()

	parser.add_argument("--input_dir",
		type=str, default=args.root_data_dir+"data_translated",)
	parser.add_argument("--output_dir",
		type=str, default=args.root_data_dir+"data_processed")
	parser.add_argument("--users_list",
		type=str, default="user_list.csv")
	parser.add_argument("--coupon_list",
		type=str, default="coupon_list_train.csv")
	parser.add_argument("--purchase_log",
		type=str, default="coupon_detail_train.csv")
	parser.add_argument("--viewing_log",
		type=str, default="coupon_visit_train.csv")
	parser.add_argument("--testing_period",
		type=int, default=7)
	parser.add_argument("--is_validation",
		action='store_false')
	args = parser.parse_args()

	split_data(
		args.input_dir,
		args.output_dir,
		args.users_list,
		args.coupon_list,
		args.purchase_log,
		args.viewing_log,
		args.testing_period,
		args.is_validation
		 )
