import numpy as np
import pandas as pd
import random
import os
import xlearn as xl
import pickle

from recutils.average_precision import mapk
from time import time
from recutils.datasets import dump_libffm_file
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from hyperopt import hp, tpe, fmin, Trials

inp_dir = "/home/ubuntu/projects/RecoTour/datasets/Ponpare/data_processed/"
train_dir = "train"
valid_dir = "valid"

# COUPONS
df_coupons_train_feat = pd.read_pickle(os.path.join(inp_dir, train_dir, 'df_coupons_train_feat.p'))
drop_cols = [c for c in df_coupons_train_feat.columns
    if ((not c.endswith('_cat')) or ('method2' in c)) and (c!='coupon_id_hash')]
df_coupons_train_cat_feat = df_coupons_train_feat.drop(drop_cols, axis=1)
coupon_categorical_cols = [c for c in df_coupons_train_cat_feat.columns if c!="coupon_id_hash"]

# USERS
df_users_train_feat = pd.read_pickle(os.path.join(inp_dir, train_dir, 'df_users_train_feat.p'))
user_categorical_cols = [c for c in df_users_train_feat.columns if c.endswith('_cat')]
user_numerical_cols = [c for c in df_users_train_feat.columns
    if ((c not in user_categorical_cols) and (c!='user_id_hash'))]

# Normalizing numerical features
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

# I want/need to ensure some order
all_cols = [c for c in df_train.columns.tolist() if c != 'interest']
cat_cols = [c for c in all_cols if c.endswith('_cat')]
num_cols = [c for c in all_cols if c not in cat_cols]
target = 'interest'
col_order=[target]+num_cols+cat_cols
df_train = df_train[col_order]

# load the validation interactions and coupon info
df_coupons_valid_feat = pd.read_pickle(os.path.join(inp_dir, 'valid', 'df_coupons_valid_feat.p'))
df_coupons_valid_cat_feat = df_coupons_valid_feat.drop(drop_cols, axis=1)

interactions_valid_dict = pickle.load(open(inp_dir + "valid/interactions_valid_dict.p", "rb"))

# Build validation data
left = pd.DataFrame({'user_id_hash':list(interactions_valid_dict.keys())})
left['key'] = 0
right = df_coupons_valid_feat[['coupon_id_hash']]
right['key'] = 0
df_valid = (pd.merge(left, right, on='key', how='outer')
    .drop('key', axis=1))
df_valid = pd.merge(df_valid, df_users_train_feat, on='user_id_hash')
df_valid = pd.merge(df_valid, df_coupons_valid_cat_feat, on = 'coupon_id_hash')
df_valid['interest'] = 0.1
df_preds = df_valid[['user_id_hash','coupon_id_hash']]
df_valid.drop(['user_id_hash','coupon_id_hash'], axis=1, inplace=True)
df_valid = df_valid[col_order]

# All needs to go to libffm format
XLEARN_DIR = inp_dir+"xlearn_data"
train_data_file = os.path.join(XLEARN_DIR,"xltrain_ffm.txt")
valid_data_file = os.path.join(XLEARN_DIR,"xlvalid_ffm.txt")
xlmodel_fname = os.path.join(XLEARN_DIR,"xlffm_model.out")
xlpreds_fname = os.path.join(XLEARN_DIR,"xlffm_preds.txt")

# catdict = {}
# for x in num_cols:
#     catdict[x] = 0
# for x in cat_cols:
#     catdict[x] = 1

# currentcode = len(num_cols)
# catcodes = {}

# currentcode_tr, catcodes_tr =  dump_libffm_file(df_train,
#     target, catdict, currentcode, catcodes, train_data_file, verbose=True)

# currentcode_va, catcodes_va =  dump_libffm_file(df_valid,
#     target, catdict, currentcode_tr, catcodes_tr, valid_data_file, verbose=True)

# # ---------------------------------------------------------------
# # WITH DEFAULTS
# params = {'epoch': 20, 'task': 'reg', 'metric': 'rmse'}
# xl_model = xl.create_ffm()
# xl_model.setTrain(train_data_file)
# xl_model.setTest(valid_data_file)
# xl_model.fit(params, xlmodel_fname)
# xl_model.predict(xlmodel_fname, xlpreds_fname)

# preds = np.loadtxt(xlpreds_fname)
# df_preds['interest'] = preds

# df_ranked = df_preds.sort_values(['user_id_hash', 'interest'],
#     ascending=[False, False])
# df_ranked = (df_ranked
#     .groupby('user_id_hash')['coupon_id_hash']
#     .apply(list)
#     .reset_index())
# recomendations_dict = pd.Series(df_ranked.coupon_id_hash.values,
#     index=df_ranked.user_id_hash).to_dict()

# actual = []
# pred = []
# for k,_ in recomendations_dict.items():
#     actual.append(list(interactions_valid_dict[k]))
#     pred.append(list(recomendations_dict[k]))

# print(mapk(actual,pred))

# ---------------------------------------------------------------
# WITH OPTIMIZATION
xlmodel_fname_tmp = os.path.join(XLEARN_DIR,"xlffm_model_tmp.out")
xlpreds_fname_tmp = os.path.join(XLEARN_DIR,"xlffm_preds_tmp.txt")

# train_data_file_opt = os.path.join(XLEARN_DIR,"xltrain_ffm_opt.txt")
# valid_data_file_opt = os.path.join(XLEARN_DIR,"xlvalid_ffm_opt.txt")

# df_train_opt, df_valid_opt = train_test_split(df_train, test_size=0.3, random_state=1981)

# catdict = {}
# for x in num_cols:
#     catdict[x] = 0
# for x in cat_cols:
#     catdict[x] = 1

# currentcode = len(num_cols)
# catcodes = {}

# currentcode_tr, catcodes_tr =  dump_libffm_file(df_train_opt,
#     target, catdict, currentcode, catcodes, train_data_file_opt, verbose=True)

# currentcode_va, catcodes_va =  dump_libffm_file(df_valid_opt,
#     target, catdict, currentcode_tr, catcodes_tr, valid_data_file_opt, verbose=True)

def xl_objective(params):

    start = time()

    xl_objective.i+=1

    params['task'] = 'reg'
    params['metric'] = 'rmse'
    params['stop_window'] = 3

    # remember hyperopt casts as floats
    params['epoch'] = int(params['epoch'])
    params['k'] = int(params['k'])

    xl_model = xl.create_ffm()
    xl_model.setTrain(train_data_file)
    # xl_model.setValidate(valid_data_file_opt)
    xl_model.setTest(valid_data_file)
    # xl_model.setQuiet()
    xl_model.fit(params, xlmodel_fname_tmp)
    xl_model.predict(xlmodel_fname_tmp, xlpreds_fname_tmp)

    preds = np.loadtxt(xlpreds_fname_tmp)
    df_preds['interest'] = preds

    df_ranked = df_preds.sort_values(['user_id_hash', 'interest'],
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

    score = mapk(actual,pred)
    end = round((time() - start)/60.,2)

    print("INFO: iteration {} was completed in {} min. Score {:.3f}".format(xl_objective.i, end, score))

    return 1-score

# xl_model = xl.create_ffm()
# xl_model.setTrain(train_data_file)
# # xl_model.setValidate(valid_data_file_opt)
# xl_model.setTest(valid_data_file)
# # xl_model.setQuiet()

xl_parameter_space = {
    'lr': hp.uniform('lr', 0.1, 0.4),
    'lambda': hp.uniform('lambda', 0.00001, 0.00005),
    'init': hp.uniform('init', 0.4, 0.8),
    'epoch': hp.quniform('epoch', 10, 20, 2),
    'k': hp.quniform('k', 4, 8, 1)
    }

trials = Trials()
xl_objective.i = 0
best_ffm = fmin(
    fn=xl_objective,
    space=xl_parameter_space,
    algo=tpe.suggest,
    max_evals=5,
    trials=trials
    )
pickle.dump(best_ffm, open(os.path.join(XLEARN_DIR,'best_ffm.p'), "wb"))
pickle.dump(trials.best_trial, open(os.path.join(XLEARN_DIR,'best_trial_ffm.p'), "wb"))

# # -----------------------------------------------------------------------------
# # WITH CV
# rnd_indx_cv = random.sample(range(df_train.shape[0]), round(df_train.shape[0]*0.1))
# df_train_cv = df_train.iloc[rnd_indx_cv, :]
# seed = random.randint(1,100)
# kf = KFold(n_splits=3, shuffle=True, random_state=seed)
# train_fpaths, valid_fpaths, valid_target_fpaths = [],[],[]
# for i, (train_index, valid_index) in enumerate(kf.split(df_train_cv)):

#     currentcode = len(num_cols)
#     catcodes = {}

#     print("INFO: iteration {} of {}".format(i+1,kf.n_splits))

#     df_tr = df_train_cv.iloc[train_index,:]
#     df_va = df_train_cv.iloc[valid_index,:]

#     train_fpath = os.path.join(XLEARN_DIR,'train_ffm_part_'+str(i)+".txt")
#     valid_fpath = os.path.join(XLEARN_DIR,'valid_ffm_part_'+str(i)+".txt")
#     valid_target_fpath = os.path.join(XLEARN_DIR,'target_ffm_part_'+str(i)+".txt")

#     print("INFO: saving libffm training file to {}".format(train_fpath))
#     currentcode_tr, catcodes_tr =  dump_libffm_file(df_tr,
#         target, catdict, currentcode, catcodes, train_fpath, verbose=True)

#     print("INFO: saving libffm validatio file to {}".format(valid_fpath))
#     currentcode_va, catcodes_va =  dump_libffm_file(df_va,
#         target, catdict, currentcode_tr, catcodes_tr, valid_fpath, verbose=True)

#     print("INFO: saving y_valid to {}".format(valid_target_fpath))
#     np.savetxt(valid_target_fpath, df_va[target].values)

#     train_fpaths.append(train_fpath)
#     valid_fpaths.append(valid_fpath)
#     valid_target_fpaths.append(valid_target_fpath)

# xl_parameter_space = {
#     'lr': hp.uniform('lr', 0.01, 0.5),
#     'lambda': hp.uniform('lambda', 0.001,0.01),
#     'init': hp.uniform('init', 0.2,0.8),
#     'epoch': hp.quniform('epoch', 10, 50, 5),
#     'k': hp.quniform('k', 4, 10, 1),
# }

# xl_model = xl.create_ffm()

# def xl_objective(params):

#     xl_objective.i+=1

#     params['task'] = 'reg'
#     params['metric'] = 'rmse'
#     params['stop_window'] = 3

#     # remember hyperopt casts as floats
#     params['epoch'] = int(params['epoch'])
#     params['k'] = int(params['k'])

#     results = []
#     for train, valid, target in zip(train_fpaths, valid_fpaths, valid_target_fpaths):

#         preds_fname = os.path.join(XLEARN_DIR, 'tmp_output.txt')
#         model_fname = os.path.join(XLEARN_DIR, "tmp_model.out")

#         xl_model.setTrain(train)
#         xl_model.setValidate(valid)
#         xl_model.setQuiet()
#         xl_model.fit(params, model_fname)

#         xl_model.setTest(valid)
#         xl_model.predict(model_fname, preds_fname)

#         y_valid = np.loadtxt(target)
#         predictions = np.loadtxt(preds_fname)
#         loss = np.sqrt(mean_squared_error(y_valid, predictions))

#         results.append(loss)

#     error = np.mean(results)
#     print("INFO: iteration {} error {:.3f}".format(xl_objective.i, error))

#     return error

# xl_model = xl.create_ffm()

# xl_objective.i = 0
# best_ffm = fmin(
#     fn=xl_objective,
#     space=xl_parameter_space,
#     algo=tpe.suggest,
#     max_evals=3
#     )

# pickle.dump(best_ffm, open(os.path.join(XLEARN_DIR,'best_ffm.p'), "wb"))