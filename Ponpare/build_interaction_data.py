import numpy as np
import pandas as pd
import os
import argparse
import pickle
import matplotlib.pyplot as plt

from scipy.sparse import lil_matrix, csr_matrix, save_npz


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


def plot_adjustment_func(x, params, xmin=None, xmax=None, func='sigmoid'):

	if not xmin: xmin = np.min(x)
	if not xmax: xmax = np.max(x)
	x = np.sort(x)

	if func == 'sigmoid':
		Z = sigmoid(x, params['xmid'], params['tau'], params['top'])
	elif func == 'linear':
		Z = combined_linear(x, params['xmid'], params['ylow'], params['ymid'], params['ytop'])

	plt.plot(x, Z, color='red', lw=0.4)
	plt.xlim((xmin, xmax))
	plt.show()


def build_interaction_df(inp_dir, out_dir, recency=False, mode=0):

	# Interactions
	df_purchases_train = pd.read_pickle(os.path.join(inp_dir, 'train', 'df_purchases_train.p'))
	df_visits_train = pd.read_pickle(os.path.join(inp_dir, 'train', 'df_visits_train.p'))
	df_visits_train.rename(index=str, columns={'view_coupon_id_hash': 'coupon_id_hash'}, inplace=True)

	# train users and coupons
	df_coupons_train_feat = pd.read_pickle(os.path.join(inp_dir, 'train', 'df_coupons_train_feat.p'))
	df_user_train_feat = pd.read_pickle(os.path.join(inp_dir, 'train', 'df_user_train_feat.p'))
	train_users = df_user_train_feat.user_id_hash.unique()
	train_coupons = df_coupons_train_feat.coupon_id_hash.unique()

	# subset activities according to the users and coupons in training
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

	if mode==0:
		df_interest.to_pickle(os.path.join(out_dir,'train','df_interest.p'))
	elif mode==1:
		return df_interest
	elif mode==2:
		df_interest.to_pickle(os.path.join(out_dir,'train','df_interest.p'))
		return df_interest


def build_interaction_mtx(inp_dir, out_dir):

	fpath = os.path.join(out_dir, 'train', 'df_interest.p')
	if os.path.isfile(fpath):
		df_interest = pd.read_pickle(fpath)
	else:
		df_interest = build_interaction_df(inp_dir, out_dir, mode=2)

	users = df_interest.user_id_hash.unique()
	items = df_interest.coupon_id_hash.unique()
	users_idx_dict = {k:v for v,k in enumerate(users)}
	items_idx_dict = {k:v for v,k in enumerate(items)}

	# lil_matrix for speed
	interactions_mtx = lil_matrix((users.shape[0], items.shape[0]))
	for i, (_,row) in enumerate(df_interest.iterrows()):
		if i%100000==0:
			print('INFO: filled {} out of {} interactions'.format(i, df_interest.shape[0]))
		user_idx = users_idx_dict[row['user_id_hash']]
		item_idx = items_idx_dict[row['coupon_id_hash']]
		interest = row['interest']
		interactions_mtx[user_idx,item_idx] = interest

	# to csr to save it (save lil format is not implemented)
	interactions_mtx = 	interactions_mtx.tocsr()
	pickle.dump(users_idx_dict, open(os.path.join(out_dir, 'train', 'users_idx_dict.p'), 'wb'))
	pickle.dump(items_idx_dict, open(os.path.join(out_dir, 'train', 'items_idx_dict.p'), 'wb'))
	save_npz(os.path.join(out_dir, 'train', "interactions_mtx.npz"), interactions_mtx)


def build_user_and_item_feat_mtx(inp_dir, out_dir):

	users_idx_dict_path = os.path.join(out_dir, 'train', 'users_idx_dict.p')
	items_idx_dict_path = os.path.join(out_dir, 'train', 'items_idx_dict.p')
	if os.path.isfile(users_idx_dict_path):
		users_idx_dict = pickle.load(open(users_idx_dict_path, 'rb'))
		items_idx_dict = pickle.load(open(items_idx_dict_path, 'rb'))
	else:
		raise FileNotFoundError("run build_interaction_mtx first")

	df_coupons_train_feat = pd.read_pickle(os.path.join(inp_dir, 'train', 'df_coupons_train_feat.p'))
	df_user_train_feat = pd.read_pickle(os.path.join(inp_dir, 'train', 'df_user_train_feat.p'))

	df_coupons_train_feat =  df_coupons_train_feat[df_coupons_train_feat.coupon_id_hash.isin(items_idx_dict.keys())]
	df_user_train_feat = df_user_train_feat[df_user_train_feat.user_id_hash.isin(users_idx_dict.keys())]

	df_coupons_order = pd.DataFrame({'coupon_id_hash': list(items_idx_dict.keys()), 'idx': list(items_idx_dict.values())})
	df_users_order = pd.DataFrame({'user_id_hash': list(users_idx_dict.keys()), 'idx': list(users_idx_dict.values())})

	df_coupons_train_feat = df_coupons_train_feat.merge(df_coupons_order, on='coupon_id_hash')
	df_user_train_feat = df_user_train_feat.merge(df_users_order, on='user_id_hash')

	df_coupons_train_feat.sort_values('idx', inplace=True)
	df_coupons_train_feat.drop(['coupon_id_hash', 'idx'], axis=1, inplace=True)
	df_coupons_train_feat = df_coupons_train_feat.astype(int)

	df_user_train_feat.sort_values('idx', inplace=True)
	df_user_train_feat.drop(['user_id_hash', 'idx'], axis=1, inplace=True)

	user_feat_mtx_colnames = df_user_train_feat.columns.tolist()
	item_feat_mtx_colnames = df_coupons_train_feat.columns.tolist()

	user_feat_mtx = csr_matrix(df_user_train_feat.values)
	item_feat_mtx = csr_matrix(df_coupons_train_feat.values)

	pickle.dump(user_feat_mtx_colnames,
		open(os.path.join(out_dir,'train', "user_feat_mtx_colnames.p"), "wb"))
	pickle.dump(item_feat_mtx_colnames,
		open(os.path.join(out_dir,'train', "item_feat_mtx_colnames.p"), "wb"))

	save_npz(os.path.join(out_dir, 'train', "user_feat_mtx.npz"), user_feat_mtx)
	save_npz(os.path.join(out_dir, 'train', "item_feat_mtx.npz"), item_feat_mtx)


if __name__ == '__main__':

	parser = argparse.ArgumentParser(description="build interaction datasets")

	parser.add_argument("--input_dir", type=str, default="../datasets/Ponpare/data_processed")
	parser.add_argument("--output_dir", type=str, default="../datasets/Ponpare/data_processed")

	args = parser.parse_args()

	build_interaction_df(args.input_dir, args.output_dir)
	build_interaction_mtx(args.input_dir, args.output_dir)
	build_user_and_item_feat_mtx(args.input_dir, args.output_dir)