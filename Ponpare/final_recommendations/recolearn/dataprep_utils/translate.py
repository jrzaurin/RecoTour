import os
import pandas as pd
import argparse


def translate(input_dir, output_dir, documentation_dir, translate_fname, prefecture_fname):

	translate_df = pd.read_excel(os.path.join(documentation_dir,translate_fname) ,skiprows=5)

	caps_col_idx = [i for i,c in enumerate(translate_df.columns) if 'CAPSULE' in c]
	engl_col_idx = [i for i,c in enumerate(translate_df.columns) if 'English' in c]

	capsule_text_df = translate_df.iloc[:, [caps_col_idx[0],engl_col_idx[0]]]
	capsule_text_df.columns = ['capsule_text', 'english_translation']

	genre_name_df = translate_df.iloc[:, [caps_col_idx[1],engl_col_idx[1]]]
	genre_name_df.columns = ['genre_name', 'english_translation']
	genre_name_df = genre_name_df[~genre_name_df.genre_name.isna()]

	# create capsule_text and genre_name dictionaries
	capsule_text_dict = dict(zip(capsule_text_df.capsule_text, capsule_text_df.english_translation))
	genre_name_dict = dict(zip(genre_name_df.genre_name, genre_name_df.english_translation))

	# create prefecture dictionary for region/area translation
	prefecture_dict = {}
	prefecture_path = os.path.join(input_dir,prefecture_fname)
	with open(prefecture_path, "r") as f:
		stuff = f.readlines()
		for line in stuff:
			line = line.rstrip().split(",")
			prefecture_dict[line[0]] = line[1]

	csv_files = []
	for _,_,files in os.walk(input_dir):
	    for file in files:
	    	if file.endswith(".csv"):
	    		csv_files.append(file)

	# define a dictionary with the columns to replace and the dictionary to
	# replace them
	replace_cols = {
		'capsule_text':'capsule_text_dict',
		'genre_name':'genre_name_dict',
		'pref_name':'prefecture_dict',
		'large_area_name':'prefecture_dict',
		'ken_name':'prefecture_dict',
		'small_area_name':'prefecture_dict'
		}

	csv_files = [c for c in csv_files if c not in ['prefecture_locations.csv','sample_submission.csv']]
	for f in csv_files:
		print("INFO: translating {} into {}".format(os.path.join(input_dir,f), os.path.join(output_dir,f)))
		tmp_df = pd.read_csv(os.path.join(input_dir,f))
		tmp_df.columns = [c.lower() for c in tmp_df]

		for col in tmp_df.columns:
			if col in replace_cols.keys():
				replace_dict = eval(replace_cols[col])
				tmp_df[col].replace(replace_dict, inplace=True)

		tmp_df.to_csv(os.path.join(output_dir,f), index=False)


if __name__ == '__main__':

	parser = argparse.ArgumentParser(description="translate data from japanese to english")

	parser.add_argument(
		"--root_data_dir",
		type=str, default="/home/ubuntu/projects/RecoTour/datasets/Ponpare/",)
	args = parser.parse_args()

	parser.add_argument("--input_dir",
		type=str, default=args.root_data_dir+"data",)
	parser.add_argument("--output_dir",
		type=str, default=args.root_data_dir+"data_translated")
	parser.add_argument("--documentation_dir",
		type=str, default=args.root_data_dir+"data/documentation")
	parser.add_argument("--translate_fname",
		type=str, default="CAPSULE_TEXT_Translation.xlsx")
	parser.add_argument("--prefecture_fname",
		type=str, default="prefecture.txt")
	args = parser.parse_args()

	translate(
		args.input_dir,
		args.output_dir,
		args.documentation_dir,
		args.translate_fname,
		args.prefecture_fname,
		)
