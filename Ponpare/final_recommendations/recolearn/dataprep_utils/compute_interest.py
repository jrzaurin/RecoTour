import numpy as np
import pandas as pd
import os
import argparse
import pickle


def interest_dataframe(work_dir, is_validation, recency):

	if is_validation:
		train_dir = os.path.join(work_dir, 'train')
	else:
		train_dir = os.path.join(work_dir, 'ftrain')

	# Interactions: purchases and visits
	df_purchases_train = pd.read_pickle(os.path.join(train_dir, 'df_purchases_train.p'))
	df_visits_train = pd.read_pickle(os.path.join(train_dir, 'df_visits_train.p'))
	df_visits_train.rename(index=str, columns={'view_coupon_id_hash': 'coupon_id_hash'}, inplace=True)

	# training users and coupons
	df_coupons_train_feat = pd.read_pickle(os.path.join(train_dir, 'df_coupons_train_feat.p'))
	df_user_train_feat = pd.read_pickle(os.path.join(train_dir, 'df_users_train_feat.p'))
	train_users = df_user_train_feat.user_id_hash.unique()
	train_coupons = df_coupons_train_feat.coupon_id_hash.unique()

	# subset activities according to the users and coupons in training. We
	# lose one customer (22624->22623) than viewed on coupon that is not in
	# training coupons and bought another one but not in training period
	df_vtr = df_visits_train[df_visits_train.user_id_hash.isin(train_users) &
		df_visits_train.coupon_id_hash.isin(train_coupons)]
	df_ptr = df_purchases_train[df_purchases_train.user_id_hash.isin(train_users) &
		df_purchases_train.coupon_id_hash.isin(train_coupons)]

	# for purchases interest will be 1
	df_interest_ptr = (df_ptr
		.groupby(['user_id_hash','coupon_id_hash'])['days_to_present']
		.min()
		.reset_index())
	df_interest_ptr['interest'] = 1.

	# remove from the visits table those pairs user-coupon that ended up in purchases
	activity_hash_p = (df_interest_ptr['user_id_hash'] + "_" +
		df_interest_ptr['coupon_id_hash']).unique()
	df_vtr['activity_hash'] = (df_vtr['user_id_hash'] + "_" +
		df_vtr['coupon_id_hash'])
	df_vtr = df_vtr[~df_vtr.activity_hash.isin(activity_hash_p)]
	df_vtr.drop('activity_hash', axis=1, inplace=True)

	# for visits will depend on number of visits
	df_vtr_coupon_views = (df_vtr
		.groupby(['user_id_hash','coupon_id_hash'])
		.size()
		.reset_index())
	df_vtr_coupon_views.columns = ['user_id_hash','coupon_id_hash','views_count']
	df_vtr_most_recent_view = (df_vtr
		.groupby(['user_id_hash','coupon_id_hash'])['days_to_present']
		.min()
		.reset_index())
	df_interest_vtr = pd.merge(df_vtr_coupon_views, df_vtr_most_recent_view,
		on=['user_id_hash','coupon_id_hash'])
	df_interest_vtr['interest'] = 0.

	# let's adjust interest for visits using a sigmoid
	vxmid, vtau, vtop = 3, 1, 0.9
	df_interest_vtr['interest'] = sigmoid(df_interest_vtr.views_count.values, vxmid, vtau, vtop)
	df_interest_vtr.drop('views_count', axis=1, inplace=True)

	# in case we want to add a factor depending on recency of purchase
	df_interest = pd.concat([df_interest_ptr, df_interest_vtr],
		axis=0, ignore_index=True)
	df_interest['days_to_present_inv'] = df_interest.days_to_present.max() - df_interest.days_to_present

	rxmid, rtau, rtop = 150, 30, 1
	df_interest['recency_factor'] = sigmoid(df_interest.days_to_present_inv.values, rxmid, rtau, rtop)
	df_interest.drop(['days_to_present', 'days_to_present_inv'], axis=1, inplace=True)
	df_interest = df_interest.sample(frac=1).reset_index(drop=True)

	if recency:
		print("INFO: interest computed taking into accoun recency")
		df_interest['interest'] = df_interest['interest'] * df_interest['recency_factor']
	else:
		print("INFO: interest computed without taking into accoun recency")

	df_interest.to_pickle(os.path.join(train_dir, 'df_interest.p'))


def sigmoid(x, xmid, tau, top):
	"""
	Sigmoid with upper limit
	"""
	return top / (1. + np.exp(-(x-xmid)/tau))


def combined_linear(x, xmid, ylow, ymid, ytop):
	"""
	Truncated straight lines
	"""
	m1 = (ymid-ylow)/(xmid)
	b1 = ylow

	x2 = np.max(x)
	m2 = (ytop-ymid)/(x2-xmid)
	b2 = ymid-(m2*xmid)

	x1_range = x[np.where(x<=xmid)[0]]
	x2_range = x[np.where(x>xmid)[0]]

	l1 = m1*x1_range + b1
	l2 = m2*x2_range + b2

	return np.hstack([l1,l2])


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
	parser.add_argument("--recency",
		action='store_true')

	args = parser.parse_args()

	interest_dataframe(
		args.work_dir,
		args.is_validation,
		args.recency
		)