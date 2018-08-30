import argparse

from recolearn.dataprep_utils.translate import translate
from recolearn.dataprep_utils.set_experiment import split_data
from recolearn.dataprep_utils.feature_engineering_items import coupon_features
from recolearn.dataprep_utils.feature_engineering_users import user_features
from recolearn.dataprep_utils.compute_interest import interest_dataframe
from recolearn.recosystem import *

if __name__ == '__main__':

	parser = argparse.ArgumentParser()

	parser.add_argument(
		"--root_data_dir",
		type=str, default="/home/ubuntu/projects/RecoTour/datasets/Ponpare/",)
	args = parser.parse_args()

	parser.add_argument("--org_data",
		type=str, default=args.root_data_dir+"data",)
	parser.add_argument("--translate_dir",
		type=str, default=args.root_data_dir+"data_translated_tmp")
	parser.add_argument("--documentation_dir",
		type=str, default=args.root_data_dir+"data/documentation")
	parser.add_argument("--processed_dir",
		type=str, default=args.root_data_dir+"data_processed_tmp")

	parser.add_argument("--translate_fname",
		type=str, default="CAPSULE_TEXT_Translation.xlsx")
	parser.add_argument("--prefecture_fname",
		type=str, default="prefecture.txt")
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
	parser.add_argument("--recency",
		action='store_true')

	args = parser.parse_args()

	if (args.is_validation):
		train_dir = "train"
	else:
		train_dir = "ftrain"

	translate(
		args.org_data,
		args.translate_dir,
		args.documentation_dir,
		args.translate_fname,
		args.prefecture_fname
		)

	split_data(
		args.translate_dir,
		args.processed_dir,
		args.users_list,
		args.coupon_list,
		args.purchase_log,
		args.viewing_log,
		args.testing_period,
		args.is_validation
		)

	coupon_features(
		args.processed_dir,
		args.is_validation
		)

	user_features(
		args.processed_dir,
		args.is_validation
		)

	interest_dataframe(
		args.processed_dir,
		args.is_validation,
		args.recency
		)

	df_purchases_test, df_visits_test, df_interest = \
	load_interactions_test_data(
		args.processed_dir,
		train_dir
		)

	# hot: customers we have seen before
	if args.is_validation:
		lgbopt = LGBOptimize(args.processed_dir)
		lgbopt.optimize(maxevals=20)

	lgbrec = LGBRec(
		args.processed_dir,
		train_dir
		)
	hot_recommendations = lgbrec.lightgbm_recommendations()
	hot_interactions = Interactions(is_hot=True)
	hot_interactions_dict = hot_interactions.interactions_dictionary(
		df_purchases_test,
		df_visits_test,
		df_interest
		)

	# cold: new customers
	mprec = MPRec(
		args.processed_dir,
		train_dir
		)
	cold_recommendations = mprec.most_popular_recommendations()
	cold_interactions = Interactions(is_hot=False)
	cold_interactions_dict = cold_interactions.interactions_dictionary(
		df_purchases_test,
		df_visits_test,
		df_interest
		)

	# cuz you're hot and your cold
	interactions_dict = cold_interactions_dict.copy()
	interactions_dict.update(hot_interactions_dict)
	recomendations_dict = cold_recommendations.copy()
	recomendations_dict.update(hot_recommendations)

	final_mapk = compute_mapk(interactions_dict, recomendations_dict)
	print("MAP: {}".format(final_mapk))