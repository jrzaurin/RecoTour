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

df_train_rn = df_train.sample(n=10000)

df_test = df_train.sample(n=1000)


currentcode = len(num_cols)
catdict = {}
catcodes = {}
for x in num_cols:
    catdict[x] = 0
for x in cat_cols:
    catdict[x] = 1

noofrows = df_train_rn.shape[0]
noofcolumns = len(df_train_rn.columns)
with open("trainffm.txt", "w") as text_file:
    for n, r in enumerate(range(noofrows)):
        if((n%100)==0):
            print('Row',n)
        datastring = ""
        datarow = df_train_rn.iloc[r].to_dict()
        datastring += str(datarow[target])

        for i, x in enumerate(catdict.keys()):
            if(catdict[x]==0):
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
        a.append(datastring)
        text_file.write(datastring)


noofrows = df_test.shape[0]
noofcolumns = len(df_test.columns)
with open("testffm.txt", "w") as text_file:
    for n, r in enumerate(range(noofrows)):
        if((n%100)==0):
            print('Row',n)
        datastring = ""
        datarow = df_test.iloc[r].to_dict()
        datastring += str(datarow[target])

        for i, x in enumerate(catdict.keys()):
            if(catdict[x]==0):
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

import xlearn as xl

ffm_model = xl.create_ffm()
ffm_model.setTrain("trainffm.txt")
ffm_model.setValidate("testffm.txt")
ffm_model.setTXTModel("model.txt")

param = {'task':'reg', # ‘binary’ for classification, ‘reg’ for Regression
         'k':10,           # Size of latent factor
         'lr':0.1,        # Learning rate for GD
         'lambda':0.0002, # L2 Regularization Parameter
         'epoch':10       # Maximum number of Epochs
        }

ffm_model.fit(param, "model.out")

