import numpy as np
import pandas as pd
import argparse
import os

# this is the way we will design the experiment:

# 1-. INTERACTIONS: we will define training and validation sets for
# interactions -> visits and purchaes

# 2-. COUPONS: based on the testing set they provide in the competition,
# testing coupons are selected based on dispfrom and are displayed for a week.
# The fact that they provided with features for the testing coupons suggest
# that we can use ALL coupons to engineer the features and the split into
# training and testing coupons. Nonetheless, we will also explore the
# possibility where new coupons have never seen before, we need to recommend
# on the fly, and there is a possibility that there are new features in those
# coupons. In this case, we would need to recommend based on some similarity
# function between new and already-seen coupons

# 3-. USERS: We will concentrate in users that have been seen during training.

def build_validation_sets(inp_dir, out_dir, tp):

	# the master list of users in the dataset
	df_users = pd.read_csv(os.path.join(inp_dir,"user_list.csv"))
	df_users['reg_date'] =  pd.to_datetime(df_users.reg_date, infer_datetime_format=True)

	# master list of coupons which are considered part of the training set
	df_coupons = pd.read_csv(os.path.join(inp_dir,"coupon_list_train.csv"))
	df_coupons['dispfrom'] = pd.to_datetime(df_coupons.dispfrom, infer_datetime_format=True)
	df_coupons['dispend'] = pd.to_datetime(df_coupons.dispend, infer_datetime_format=True)
	df_coupons['validfrom'] = pd.to_datetime(df_coupons.validfrom, infer_datetime_format=True)
	df_coupons['validend'] = pd.to_datetime(df_coupons.validend, infer_datetime_format=True)

	# the viewing log of users browsing coupons during training
	df_visits = pd.read_csv(os.path.join(inp_dir,"coupon_visit_train.csv"))
	df_visits['i_date'] = pd.to_datetime(df_visits.i_date, infer_datetime_format=True)

	# the purchase log of users buying coupons during training
	df_purchases = pd.read_csv(os.path.join(inp_dir,"coupon_detail_train.csv"))
	df_purchases['i_date'] = pd.to_datetime(df_purchases.i_date, infer_datetime_format=True)

	# Here we want to explore different recommendation algorithms. Therefore,
	# for the purposes of this excercise we can ignore the 310 coupons that
	# are provided in the kaggle competition as part of the testing set, since
	# we want to know how good are we recommending. We will divide the time
	# period in train, validation and testing
	df_interactions = [df_visits, df_purchases]
	most_recent = []
	for df in df_interactions:
		for col in df.columns:
			if col == 'i_date':
				most_recent.append(df[col].max())
	present = np.max(most_recent)

	tmp_df_visits = pd.DataFrame({'present': [present]*df_visits.shape[0]})
	df_visits['days_to_present'] = (tmp_df_visits['present'] - df_visits['i_date'])
	df_visits['days_to_present'] = df_visits.days_to_present.dt.days

	tmp_df_detail = pd.DataFrame({'present': [present]*df_purchases.shape[0]})
	df_purchases['days_to_present'] = (tmp_df_detail['present'] - df_purchases['i_date'])
	df_purchases['days_to_present'] = df_purchases.days_to_present.dt.days

	# In reality, for most examples, we will just use users seen in training
	tmp_df_users = pd.DataFrame({'present': [present]*df_users.shape[0]})
	df_users['days_to_present'] = (tmp_df_users['present'] - df_users['reg_date'])
	df_users['days_to_present'] = df_users.days_to_present.dt.days

	tmp_df_coupons = pd.DataFrame({'present': [present]*df_coupons.shape[0]})
	df_coupons['days_to_present'] = (tmp_df_detail['present'] - df_coupons['dispfrom'])
	df_coupons['days_to_present'] = df_coupons.days_to_present.dt.days

	del(tmp_df_visits,tmp_df_detail,tmp_df_users,tmp_df_coupons)

	# We will explore a series of scenarios that will be discussed when we get
	# there, but just for convenience, let's split all datasets
	df_visits['days_to_present_flag'] = df_visits.days_to_present.apply(
		lambda x: 0 if x<=tp-1 else 1 if ((x>tp-1) and (x<=(tp*2)-1)) else 2)
	df_purchases['days_to_present_flag'] = df_purchases.days_to_present.apply(
		lambda x: 0 if x<=tp-1 else 1 if ((x>tp-1) and (x<=(tp*2)-1)) else 2)
	df_users['days_to_present_flag'] = df_users.days_to_present.apply(
		lambda x: 0 if x<=tp-1 else 1 if ((x>tp-1) and (x<=(tp*2)-1)) else 2)
	df_coupons['days_to_present_flag'] = df_coupons.days_to_present.apply(
		lambda x: 0 if x<=tp-1 else 1 if ((x>tp-1) and (x<=(tp*2)-1)) else 2)

	df_l = ['df_visits', 'df_purchases', 'df_users', 'df_coupons']
	for df in df_l:
		print('INFO: splitting {}'.format(df.split('_')[1]))
		tmp_train = eval(df)[eval(df)['days_to_present_flag'] == 2]
		tmp_valid = eval(df)[eval(df)['days_to_present_flag'] == 1]
		tmp_test  = eval(df)[eval(df)['days_to_present_flag'] == 0]

		tmp_train.drop('days_to_present_flag', axis=1, inplace=True)
		tmp_valid.drop('days_to_present_flag', axis=1, inplace=True)
		tmp_test.drop('days_to_present_flag', axis=1, inplace=True)

		tmp_train.to_pickle(open(os.path.join(out_dir,'train',df+'_train.p'), 'wb'))
		tmp_valid.to_pickle(open(os.path.join(out_dir,'valid',df+'_valid.p'), 'wb'))
		tmp_test.to_pickle(open(os.path.join(out_dir,'test',df+'_test.p'), 'wb'))

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description="create validation sets based on a given testing period")

	parser.add_argument("--input_dir", type=str, default="../datasets/Ponpare/data_translated",)
	parser.add_argument("--output_dir", type=str, default="../datasets/Ponpare/data_processed")
	parser.add_argument("--testing_period", type=int, default=7)
	args = parser.parse_args()

	build_validation_sets(args.input_dir, args.output_dir, args.testing_period)
