import numpy as np
import pandas as pd
import random
import os
import xlearn as xl
import pickle

from recutils.average_precision import mapk
from recutils.datasets import dump_libffm_file
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from hyperopt import hp, tpe
from hyperopt.fmin import fmin

inp_dir = "../datasets/Ponpare/data_processed/"
train_dir = "train"
valid_dir = "valid"

# train coupon features
df_coupons_train_feat = pd.read_pickle(os.path.join(inp_dir, train_dir, 'df_coupons_train_feat.p'))
drop_cols = [c for c in df_coupons_train_feat.columns
	if (('_cat' not in c) or ('method2' in c)) and (c!='coupon_id_hash')]
# for coupons all are categorical
df_coupons_train_cat_feat = df_coupons_train_feat.drop(drop_cols, axis=1)
coupon_categorical_cols = [c for c in df_coupons_train_cat_feat.columns if c!="coupon_id_hash"]

# train user features: there are a lot of features for users, both, numerical
# and categorical. We keep them all
df_users_train_feat = pd.read_pickle(os.path.join(inp_dir, train_dir, 'df_user_train_feat.p'))
user_categorical_cols = [c for c in df_users_train_feat.columns if c.endswith('_cat')]
user_numerical_cols = [c for c in df_users_train_feat.columns
    if ((c not in user_categorical_cols) and (c!='user_id_hash'))]

# a bit of prepocessing for the numerical features. Not needed, but I tend to do these things...
user_numerical_df = df_users_train_feat[user_numerical_cols]
user_numerical_df_norm = (user_numerical_df-user_numerical_df.min())/(user_numerical_df.max()-user_numerical_df.min())
df_users_train_feat.drop(user_numerical_cols, axis=1, inplace=True)
df_users_train_feat = pd.concat([user_numerical_df_norm, df_users_train_feat], axis=1)

# interest dataframe
df_interest = pd.read_pickle(os.path.join(inp_dir, train_dir, 'df_interest.p'))
df_train = pd.merge(df_interest, df_users_train_feat, on='user_id_hash')
df_train = pd.merge(df_train, df_coupons_train_cat_feat, on = 'coupon_id_hash')

# for the time being we ignore recency
df_train.drop(['user_id_hash','coupon_id_hash','recency_factor'], axis=1, inplace=True)

all_cols = [c for c in df_train.columns.tolist() if c != 'interest']
cat_cols = [c for c in all_cols if c.endswith('_cat')]
num_cols = [c for c in all_cols if c not in cat_cols]
target = 'interest'
col_order=[target]+num_cols+cat_cols
df_train = df_train[col_order]

# Let's set a similar experiment to the one in xlearn_fm_linear
XLEARN_DIR = "xlearn_data"

catdict = {}
for x in num_cols:
    catdict[x] = 0
for x in cat_cols:
    catdict[x] = 1

rnd_indx_cv = random.sample(range(df_train.shape[0]), round(df_train.shape[0]*0.1))
df_train_cv = df_train.iloc[rnd_indx_cv, :]
seed = random.randint(1,100)
kf = KFold(n_splits=3, shuffle=True, random_state=seed)
train_fpaths, valid_fpaths, valid_target_fpaths = [],[],[]
for i, (train_index, valid_index) in enumerate(kf.split(df_train_cv)):

    currentcode = len(num_cols)
    catcodes = {}

    print("INFO: iteration {} of {}".format(i+1,kf.n_splits))

    df_tr = df_train_cv.iloc[train_index,:]
    df_va = df_train_cv.iloc[valid_index,:]

    train_fpath = os.path.join(XLEARN_DIR,'train_ffm_part_'+str(i)+".txt")
    valid_fpath = os.path.join(XLEARN_DIR,'valid_ffm_part_'+str(i)+".txt")
    valid_target_fpath = os.path.join(XLEARN_DIR,'target_ffm_part_'+str(i)+".txt")

    print("INFO: saving libffm training file to {}".format(train_fpath))
    currentcode_tr, catcodes_tr =  dump_libffm_file(df_tr,
        target, catdict, currentcode, catcodes, train_fpath, verbose=True)

    print("INFO: saving libffm validatio file to {}".format(valid_fpath))
    currentcode_va, catcodes_va =  dump_libffm_file(df_va,
        target, catdict, currentcode_tr, catcodes_tr, valid_fpath, verbose=True)

    print("INFO: saving y_valid to {}".format(valid_target_fpath))
    np.savetxt(valid_target_fpath, df_va[target].values)

    train_fpaths.append(train_fpath)
    valid_fpaths.append(valid_fpath)
    valid_target_fpaths.append(valid_target_fpath)

xl_parameter_space = {
    'lr': hp.uniform('lr', 0.01, 0.5),
    'lambda': hp.uniform('lambda', 0.001,0.01),
    'init': hp.uniform('init', 0.2,0.8),
    'epoch': hp.quniform('epoch', 10, 50, 5),
    'k': hp.quniform('k', 2, 10, 1),
}

def xl_objective(params):

    xl_objective.i+=1

    params['task'] = 'reg'
    params['metric'] = 'rmse'
    params['stop_window'] = 5

    # remember hyperopt casts as floats
    params['epoch'] = int(params['epoch'])
    params['k'] = int(params['k'])

    xl_model = xl.create_ffm()

    results = []
    for train, valid, target in zip(train_fpaths, valid_fpaths, valid_target_fpaths):

        preds_fname = os.path.join(XLEARN_DIR, 'tmp_output.txt')
        model_fname = os.path.join(XLEARN_DIR, "tmp_model.out")

        xl_model.setTrain(train)
        xl_model.setValidate(valid)
        xl_model.setQuiet()
        xl_model.fit(params, model_fname)

        xl_model.setTest(valid)
        xl_model.predict(model_fname, preds_fname)

        y_valid = np.loadtxt(target)
        predictions = np.loadtxt(preds_fname)
        loss = np.sqrt(mean_squared_error(y_valid, predictions))

        results.append(loss)

    error = np.mean(results)
    print("INFO: iteration {} error {:.3f}".format(xl_objective.i, error))

    return error

xl_objective.i = 0
best_ffm = fmin(
    fn=xl_objective,
    space=xl_parameter_space,
    algo=tpe.suggest,
    max_evals=10
    )
pickle.dump(best_ffm, open(os.path.join(XLEARN_DIR,'best_ffm.p'), "wb"))

# validation activities
df_purchases_valid = pd.read_pickle(os.path.join(inp_dir, 'valid', 'df_purchases_valid.p'))
df_visits_valid = pd.read_pickle(os.path.join(inp_dir, 'valid', 'df_visits_valid.p'))
df_visits_valid.rename(index=str, columns={'view_coupon_id_hash': 'coupon_id_hash'}, inplace=True)

# Read the validation coupon features
df_coupons_valid_feat = pd.read_pickle(os.path.join(inp_dir, 'valid', 'df_coupons_valid_feat.p'))
df_coupons_valid_cat_feat = df_coupons_valid_feat.drop(drop_cols, axis=1)

# subset users that were seeing in training
train_users = df_interest.user_id_hash.unique()
df_vva = df_visits_valid[df_visits_valid.user_id_hash.isin(train_users)]
df_pva = df_purchases_valid[df_purchases_valid.user_id_hash.isin(train_users)]

# interactions in validation: here we will not treat differently purchases or
# viewed. If we recommend and it was viewed or purchased, we will considered
# it as a hit
id_cols = ['user_id_hash', 'coupon_id_hash']
df_interactions_valid = pd.concat([df_pva[id_cols], df_vva[id_cols]], ignore_index=True)
df_interactions_valid = (df_interactions_valid.groupby('user_id_hash')
    .agg({'coupon_id_hash': 'unique'})
    .reset_index())
tmp_valid_dict = pd.Series(df_interactions_valid.coupon_id_hash.values,
    index=df_interactions_valid.user_id_hash).to_dict()

valid_coupon_ids = df_coupons_valid_feat.coupon_id_hash.values
keep_users = []
for user, coupons in tmp_valid_dict.items():
    if np.intersect1d(valid_coupon_ids, coupons).size !=0:
        keep_users.append(user)
# out of 6923, we end up with 6070, so not bad
interactions_valid_dict = {k:v for k,v in tmp_valid_dict.items() if k in keep_users}

# Take the 358 validation coupons and the 6070 users seen in training and during
# validation and rank!
left = pd.DataFrame({'user_id_hash':list(interactions_valid_dict.keys())})
left['key'] = 0
right = df_coupons_valid_feat[['coupon_id_hash']]
right['key'] = 0
df_valid = (pd.merge(left, right, on='key', how='outer')
    .drop('key', axis=1))
df_valid = pd.merge(df_valid, df_users_train_feat, on='user_id_hash')
df_valid = pd.merge(df_valid, df_coupons_valid_cat_feat, on = 'coupon_id_hash')
df_valid['interest'] = 0.1
valid_col_order = ['user_id_hash','coupon_id_hash'] + col_order
df_valid = df_valid[valid_col_order]

df_user_coupon_interaction = df_valid[['user_id_hash','coupon_id_hash']]
df_valid.drop(['user_id_hash','coupon_id_hash'], axis=1, inplace=True)

train_data_file = os.path.join(XLEARN_DIR,"xltrain_ffm.txt")
valid_data_file = os.path.join(XLEARN_DIR,"xlvalid_ffm.txt")

currentcode = len(num_cols)
catcodes = {}
currentcode_tr, catcodes_tr =  dump_libffm_file(df_train,
    target, catdict, currentcode, catcodes, train_data_file, verbose=True)

currentcode_va, catcodes_va =  dump_libffm_file(df_valid,
    target, catdict, currentcode_tr, catcodes_tr, valid_data_file, verbose=True)

# optimise takes A LONG time, so for now, mostly defaults
best_ffm = {'epoch': 20, 'task': 'reg', 'metric': 'rmse', 'stop_window': 5}

xlmodel_fname = os.path.join(XLEARN_DIR,"xlffm_model.out")
xlpreds_fname = os.path.join(XLEARN_DIR,"xlffm_preds.txt")

xl_model = xl.create_ffm()
xl_model.setTrain(train_data_file)
xl_model.setTest(valid_data_file)

xl_model.fit(best_ffm, xlmodel_fname)
xl_model.predict(xlmodel_fname, xlpreds_fname)

preds = np.loadtxt(xlpreds_fname)
df_user_coupon_interaction['interest'] = preds

df_ranked = df_user_coupon_interaction.sort_values(['user_id_hash', 'interest'],
    ascending=[False, False])
df_ranked = (df_ranked
    .groupby('user_id_hash')['coupon_id_hash']
    .apply(list)
    .reset_index())
recomendations_dict = pd.Series(df_ranked.coupon_id_hash.values,
    index=df_ranked.user_id_hash).to_dict()

actual = []
pred = []
for k,_ in recomendations_dict.items():
    actual.append(list(interactions_valid_dict[k]))
    pred.append(list(recomendations_dict[k]))

print(mapk(actual,pred))