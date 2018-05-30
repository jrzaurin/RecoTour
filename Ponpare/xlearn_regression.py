import numpy as np
import pandas as pd
import random
import os

from recutils.average_precision import mapk

inp_dir = "../datasets/Ponpare/data_processed/"
train_dir = "train"
valid_dir = "valid"

# train coupon features
df_coupons_train_feat = pd.read_pickle(os.path.join(inp_dir, train_dir, 'df_coupons_train_feat.p'))

drop_cols = [c for c in df_coupons_train_feat.columns
	if (('_cat' not in c) or ('method2' in c)) and (c!='coupon_id_hash')]
df_coupons_train_cat_feat = df_coupons_train_feat.drop(drop_cols, axis=1)

# train user features: there are a lot of features for users, both, numerical
# and categorical. We keep them all
df_users_train_feat = pd.read_pickle(os.path.join(inp_dir, train_dir, 'df_user_train_feat.p'))

# interest dataframe
df_interest = pd.read_pickle(os.path.join(inp_dir, train_dir, 'df_interest.p'))

df_train = pd.merge(df_interest, df_users_train_feat, on='user_id_hash')
df_train = pd.merge(df_train, df_coupons_train_cat_feat, on = 'coupon_id_hash')

# for the time being we ignore recency
df_train.drop(['user_id_hash','coupon_id_hash','recency_factor'], axis=1, inplace=True)
# train = df_train.drop('interest', axis=1)
# y_train = df_train.interest
all_cols = [c for c in df_train.columns.tolist() if c != 'interest']
cat_cols = [c for c in df_train.columns if '_cat' in c]
num_cols = [c for c in all_cols if c not in cat_cols]
target = 'interest'
col_order=[target]+num_cols+cat_cols
df_train.columns = col_order

currentcode = len(num_cols)
catdict = {}
catcodes = {}
for x in num_cols:
    catdict[x] = 0
for x in cat_cols:
    catdict[x] = 1

noofrows = df_train.shape[0]
noofcolumns = len(df_train.columns)
with open("alltrainffm.txt", "w") as text_file:
    for n, r in enumerate(range(3)):
        if((n%100000)==0):
            print('Row',n)
        datastring = ""
        datarow = df_train.iloc[r].to_dict()
        datastring += str(datarow[target])

        for i, x in enumerate(catdict.keys()):
            if(catdict[x]==0):
            	print(currentcode)
                datastring = datastring + " "+str(i)+":"+ str(i)+":"+ str(datarow[x])
            else:
                if(x not in catcodes):
                    catcodes[x] = {}
                    currentcode +=1
                    catcodes[x][datarow[x]] = currentcode
                elif(datarow[x] not in catcodes[x]):
                    currentcode +=1
                    catcodes[x][datarow[x]] = currentcode

                code = catcodes[x][datarow[x]]
                datastring = datastring + " "+str(i)+":"+ str(int(code))+":1"
        datastring += '\n'
        text_file.write(datastring)



def convert_to_ffm(df,type,numerics,categories,features):
    currentcode = len(numerics)
    catdict = {}
    catcodes = {}
    # Flagging categorical and numerical fields
    for x in numerics:
         catdict[x] = 0
    for x in categories:
         catdict[x] = 1

    nrows = df.shape[0]
    ncolumns = len(features)
    with open(str(type) + "_ffm.txt", "w") as text_file:

    # Looping over rows to convert each row to libffm format
    for n, r in enumerate(range(nrows)):
         datastring = ""
         datarow = df.iloc[r].to_dict()
         datastring += str(int(datarow['Label']))
         # For numerical fields, we are creating a dummy field here
         for i, x in enumerate(catdict.keys()):
             if(catdict[x]==0):
                 datastring = datastring + " "+str(i)+":"+ str(i)+":"+ str(datarow[x])
             else:
         # For a new field appearing in a training example
                 if(x not in catcodes):
                     catcodes[x] = {}
                     currentcode +=1
                     catcodes[x][datarow[x]] = currentcode #encoding the feature
         # For already encoded fields
                 elif(datarow[x] not in catcodes[x]):
                     currentcode +=1
                     catcodes[x][datarow[x]] = currentcode #encoding the feature
                 code = catcodes[x][datarow[x]]
                 datastring = datastring + " "+str(i)+":"+ str(int(code))+":1"

         datastring += '\n'
         text_file.write(datastring)