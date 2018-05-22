import os
import translate as trans
import create_validation as val
import preprocessing_submission as prep_sub 
import preprocessing_validation as prep_val

if __name__ == "__main__":

	#Assert files have been properly downloaded
	for file_name in ["coupon_list_train", "coupon_list_test", 
						"coupon_detail_train", "coupon_visit_train", "user_list"]:
		assert(os.path.isfile("../Data/Data_japanese/%s.csv" % file_name))

	#Check directories have been properly set up
	if not os.path.exists("../Data/Data_translated/"):
		os.makedirs("../Data/Data_translated/")
	if not os.path.exists("../Data/Validation/"):
		os.makedirs("../Data/Validation/")
	if not os.path.exists("../Submissions/"):
		os.makedirs("../Submissions/")

	#Translate files to English
	trans.translate()

	############################################################
	#Process the data (details in preprocessing_submission.py)
	############################################################

	#Preprocessing for similarity based recommander system
	prep_sub.skim_visit_file()
	prep_sub.create_dict_user_list()
	prep_sub.create_LabelEncoded_files()
	prep_sub.create_LabelBinarized_files()
	#Define the choice of feature engineering by setting var_choice 
	var_choice = "1"
	prep_sub.prepare_similarity_data(var_choice)
	#Preprocessing for a hybrid matrix factorisation method
	prep_sub.build_biclass_user_item_mtrx()
	prep_sub.build_user_feature_matrix()
	prep_sub.build_item_feature_matrix()

    ##########################
    #Create validation sets
    ##########################
	val.create_validation_set([2012,06,17], [2012, 06, 23], "week52")
	val.create_validation_set([2012,06,10], [2012, 06, 16], "week51")

	#Create more validation sets if needed
	# create_validation_set([2012,06,3], [2012, 06, 9], "week50")
	# create_validation_set([2012,05,27], [2012, 06, 2], "week49")
	# create_validation_set([2012,05,20], [2012, 05, 26], "week48")
	
	############################################################
	#Preprocess the validation data (details in preprocessing_validation.py)
	############################################################
	
	for week_ID in ["week51", "week52"]:
		if not os.path.exists("../Data/Validation/" + week_ID + "/"):
			os.makedirs("../Data/Validation/" + week_ID + "/")
		#Preprocessing for similarity based recommander system
		prep_val.create_LabelEncoded_files(week_ID)
		prep_val.create_LabelBinarized_files(week_ID)
		var_choice = "1"
		prep_val.prepare_similarity_data(var_choice, week_ID)
		#Preprocessing for a hybrid matrix factorisation method
		prep_val.build_biclass_user_item_mtrx(week_ID)
		prep_val.build_user_feature_matrix(week_ID)
		prep_val.build_item_feature_matrix(week_ID)