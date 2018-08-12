import numpy as np
import pandas as pd
import random
import gc
import os
import xlearn as xl
import pickle

from recutils.average_precision import mapk
from sklearn.datasets import dump_svmlight_file, load_svmlight_file
from scipy.sparse import csr_matrix, save_npz
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from hyperopt import hp, tpe
from hyperopt.fmin import fmin

def xl_objective(params, method="fm"):

    xl_objective.i+=1

    params['task'] = 'reg'
    params['metric'] = 'rmse'

    # remember hyperopt casts as floats
    params['epoch'] = int(params['epoch'])
    params['k'] = int(params['k'])

    if method is "linear":
        xl_model = xl.create_linear()
    elif method is "fm":
        xl_model = xl.create_fm()

    results = []
    for train, valid, target in zip(train_fpaths, valid_fpaths, valid_target_fpaths):

        preds_fname = os.path.join(XLEARN_DIR, 'tmp_output.txt')
        model_fname = os.path.join(XLEARN_DIR, "tmp_model.out")

        xl_model.setTrain(train)
        xl_model.setTest(valid)
        xl_model.setQuiet()
        xl_model.fit(params, model_fname)
        xl_model.predict(model_fname, preds_fname)

        y_valid = np.loadtxt(target)
        predictions = np.loadtxt(preds_fname)
        loss = np.sqrt(mean_squared_error(y_valid, predictions))

        results.append(loss)

    error = np.mean(results)
    print("INFO: iteration {} error {:.3f}".format(xl_objective.i, error))

    return error

inp_dir = "../datasets/Ponpare/data_processed/"
train_dir = "train"
valid_dir = "valid"

# COUPONS
# train coupon features
df_coupons_train_feat = pd.read_pickle(os.path.join(inp_dir, train_dir, 'df_coupons_train_feat.p'))
drop_cols = [c for c in df_coupons_train_feat.columns
    if (('_cat' not in c) or ('method2' in c)) and (c!='coupon_id_hash')]
df_coupons_train_cat_feat = df_coupons_train_feat.drop(drop_cols, axis=1)
coupons_cols_to_oh = [c for c in df_coupons_train_cat_feat.columns if (c!='coupon_id_hash')]

# We are going to use FM (and linear) methods with xlearn. Since there no
# "automatic" treatment of categorical features, we need to one-hot encode
# them. The one-hot encoding process needs to be done all at once, validation
# and training datasets

# Read the validation coupon features
df_coupons_valid_feat = pd.read_pickle(os.path.join(inp_dir, 'valid', 'df_coupons_valid_feat.p'))
df_coupons_valid_cat_feat = df_coupons_valid_feat.drop(drop_cols, axis=1)
df_coupons_train_cat_feat['is_valid'] = 0
df_coupons_valid_cat_feat['is_valid'] = 1
df_all_coupons = (df_coupons_train_cat_feat
    .append(df_coupons_valid_cat_feat, ignore_index=True))
df_all_coupons_oh_feat = pd.get_dummies(df_all_coupons, columns=coupons_cols_to_oh)
df_coupons_train_oh_feat = (df_all_coupons_oh_feat[df_all_coupons_oh_feat.is_valid==0]
    .drop('is_valid', axis=1))
df_coupons_valid_oh_feat = (df_all_coupons_oh_feat[df_all_coupons_oh_feat.is_valid==1]
    .drop('is_valid', axis=1))

# USERS
df_users_train_feat = pd.read_pickle(os.path.join(inp_dir, train_dir, 'df_user_train_feat.p'))

# Here a bit or preprocessing for the numerical features
user_categorical_cols = [c for c in df_users_train_feat.columns if c.endswith('_cat')]
user_numerical_cols = [c for c in df_users_train_feat.columns
    if ((c not in user_categorical_cols) and (c!='user_id_hash'))]
user_numerical_df = df_users_train_feat[user_numerical_cols]
user_numerical_df_norm = (user_numerical_df-user_numerical_df.min())/(user_numerical_df.max()-user_numerical_df.min())
df_users_train_feat.drop(user_numerical_cols, axis=1, inplace=True)
df_users_train_feat = pd.concat([user_numerical_df_norm, df_users_train_feat], axis=1)
df_users_train_oh_feat = pd.get_dummies(df_users_train_feat, columns=user_categorical_cols)

# INTEREST
df_interest = pd.read_pickle(os.path.join(inp_dir, train_dir, 'df_interest.p'))
df_train = pd.merge(df_interest, df_users_train_oh_feat, on='user_id_hash')
df_train = pd.merge(df_train, df_coupons_train_oh_feat, on = 'coupon_id_hash')

# drop unneccesary columns
df_train.drop(['user_id_hash','coupon_id_hash','recency_factor'], axis=1, inplace=True)
y_train = df_train.interest.values
df_train.drop('interest', axis=1, inplace=True)

# Due to a series of problems when using xlearn I initially decided to run my own cv

# 1-. I will take X% of the training data and split it in 3 folds
    # the reason of subsampling is mostly due to computing time
# 2-. Save them to disk in respective files
# 3-. perform cv manually within an hyperopt function
XLEARN_DIR = inp_dir + "xlearn_data"
rnd_indx_cv = random.sample(range(df_train.shape[0]), round(df_train.shape[0]*0.1))
X_train_cv = csr_matrix(df_train.iloc[rnd_indx_cv,:].values)
y_train_cv =  y_train[rnd_indx_cv]
seed = random.randint(1,100)
kf = KFold(n_splits=3, shuffle=True, random_state=seed)
train_fpaths, valid_fpaths, valid_target_fpaths = [],[],[]
for i, (train_index, valid_index) in enumerate(kf.split(X_train_cv)):

    print("INFO: iteration {} of {}".format(i+1,kf.n_splits))

    x_tr, y_tr = X_train_cv[train_index], y_train_cv[train_index]
    x_va, y_va = X_train_cv[valid_index], y_train_cv[valid_index]

    train_fpath = os.path.join(XLEARN_DIR,'train_part_'+str(i)+".txt")
    valid_fpath = os.path.join(XLEARN_DIR,'valid_part_'+str(i)+".txt")
    valid_target_fpath = os.path.join(XLEARN_DIR,'target_part_'+str(i)+".txt")

    print("INFO: saving svmlight training file to {}".format(train_fpath))
    dump_svmlight_file(x_tr, y_tr, train_fpath)

    print("INFO: saving svmlight validatio file to {}".format(valid_fpath))
    dump_svmlight_file(x_va, y_va, valid_fpath)

    print("INFO: saving y_valid to {}".format(valid_target_fpath))
    np.savetxt(valid_target_fpath, y_va)

    train_fpaths.append(train_fpath)
    valid_fpaths.append(valid_fpath)
    valid_target_fpaths.append(valid_target_fpath)


xl_parameter_space = {
    'lr': hp.uniform('lr', 0.01, 0.5),
    'lambda': hp.uniform('lambda', 0.001,0.01),
    'init': hp.uniform('init', 0.2,0.8),
    'epoch': hp.quniform('epoch', 10, 200, 10),
    'k': hp.quniform('k', 2, 10, 1),
}

# liblinear fit
partial_objective = lambda params: xl_objective(
    params,
    method="linear")
xl_objective.i = 0
best_linear = fmin(
    fn=partial_objective,
    space=xl_parameter_space,
    algo=tpe.suggest,
    max_evals=10
    )
pickle.dump(best_linear, open(os.path.join(XLEARN_DIR,'best_linear.p'), "wb"))

# libfm fit
partial_objective = lambda params: xl_objective(
    params,
    method="fm")
xl_objective.i = 0
best_fm = fmin(
    fn=partial_objective,
    space=xl_parameter_space,
    algo=tpe.suggest,
    max_evals=10
    )
pickle.dump(best_fm, open(os.path.join(XLEARN_DIR,'best_fm.p'), "wb"))

# Read validation interactions dataset
interactions_valid_dict = pickle.load(
    open("../datasets/Ponpare/data_processed/valid/interactions_valid_dict.p", "rb"))

# Take the validation coupons and train users seen in training and during
# validation and rank!
left = pd.DataFrame({'user_id_hash':list(interactions_valid_dict.keys())})
left['key'] = 0
right = df_coupons_valid_feat[['coupon_id_hash']]
right['key'] = 0
df_valid = (pd.merge(left, right, on='key', how='outer')
    .drop('key', axis=1))
df_valid = pd.merge(df_valid, df_users_train_oh_feat, on='user_id_hash')
df_valid = pd.merge(df_valid, df_coupons_valid_oh_feat, on = 'coupon_id_hash')

train_data_file = os.path.join(XLEARN_DIR,"xltrain.txt")
valid_data_file = os.path.join(XLEARN_DIR,"xlvalid.txt")
xlmodel_fname = os.path.join(XLEARN_DIR,"xllinear_model.out")
xlpreds_fname = os.path.join(XLEARN_DIR,"xllinear_preds.txt")

X_train = csr_matrix(df_train.values)
%time dump_svmlight_file(X_train,y_train,train_data_file)
del(X_train)

X_valid = csr_matrix(df_valid
    .drop(['user_id_hash','coupon_id_hash'], axis=1)
    .values)
y_valid = np.array([0.1]*X_valid.shape[0])
%time dump_svmlight_file(X_valid,y_valid,valid_data_file)
del(X_valid)

best_param = pickle.load(open(os.path.join(XLEARN_DIR,'best_fm.p'), "rb"))
best_param['epoch'] = int(best_param['epoch'])
best_param['k'] = int(best_param['k'])
best_param['task'] = 'reg'
best_param['metric'] = 'rmse'

best_param = {'task':'reg',
            'lr':0.01,
            'lambda': 0.05,
            'epoch':10,
            'k':20}

xl_model = xl.create_fm()
xl_model.setTrain(train_fpath)
xl_model.setTest(valid_fpath)
xl_model.fit(best_param, xlmodel_fname_tmp)
xl_model.predict(xlmodel_fname_tmp, xlpreds_fname_tmp)

preds = np.loadtxt(xlpreds_fname_tmp)
df_preds = df_valid[['user_id_hash','coupon_id_hash']]
df_preds['interest'] = preds

df_ranked = df_preds.sort_values(['user_id_hash', 'interest'], ascending=[False, False])
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
