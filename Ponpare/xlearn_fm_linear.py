import numpy as np
import pandas as pd
import random
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

inp_dir = "../datasets/Ponpare/data_processed/"
train_dir = "train"
valid_dir = "valid"

# train coupon features
df_coupons_train_feat = pd.read_pickle(os.path.join(inp_dir, train_dir, 'df_coupons_train_feat.p'))
drop_cols = [c for c in df_coupons_train_feat.columns
    if (('_cat' not in c) or ('method2' in c)) and (c!='coupon_id_hash')]
df_coupons_train_cat_feat = df_coupons_train_feat.drop(drop_cols, axis=1)
coupons_cols_to_oh = [c for c in df_coupons_train_cat_feat.columns if (c!='coupon_id_hash')]

# because we are going to use linear models and FM with xlearn, there is no
# "automatic" treatment of categorical features, therefore, we one- hot encode
# them. To one hot encode we need to do it all at once, validation and
# training coupons

# Read the validation coupon features
df_coupons_valid_feat = pd.read_pickle(os.path.join(inp_dir, 'valid', 'df_coupons_valid_feat.p'))
df_coupons_valid_cat_feat = df_coupons_valid_feat.drop(drop_cols, axis=1)

df_coupons_train_cat_feat['is_valid'] = 0
df_coupons_valid_cat_feat['is_valid'] = 1

df_all_coupons = (df_coupons_train_cat_feat
    .append(df_coupons_valid_cat_feat, ignore_index=True))

# with coupons all features are categorical
coupon_categorical_cols = [c for c in df_all_coupons.columns if c.endswith('_cat')]

df_all_coupons_oh_feat = pd.get_dummies(df_all_coupons, columns=coupons_cols_to_oh)
df_coupons_train_oh_feat = (df_all_coupons_oh_feat[df_all_coupons_oh_feat.is_valid==0]
    .drop('is_valid', axis=1))
df_coupons_valid_oh_feat = (df_all_coupons_oh_feat[df_all_coupons_oh_feat.is_valid==1]
    .drop('is_valid', axis=1))

# train user features: there are a lot of features for users, both, numerical
# and categorical. We keep them all
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

# interest dataframe
df_interest = pd.read_pickle(os.path.join(inp_dir, train_dir, 'df_interest.p'))
df_train = pd.merge(df_interest, df_users_train_oh_feat, on='user_id_hash')
df_train = pd.merge(df_train, df_coupons_train_oh_feat, on = 'coupon_id_hash')

# drop unneccesary columns
df_train.drop(['user_id_hash','coupon_id_hash','recency_factor'], axis=1, inplace=True)
y_train = df_train.interest.values
df_train.drop('interest', axis=1, inplace=True)

# so, now, let's go through the issues one runs into when using this package.
# I normally prefer to use native methods rather than wrappers, but this time,
# after reading the documentation I though that the best thing would be using
# the sklearn API. For convenience, let's sample 10000 rows from the training
# dataset and consider it train dataset and an additional 1000 and consider them
# test
rnd_indx = random.sample(range(df_train.shape[0]), 10000)
tmp_X_train = df_train.iloc[rnd_indx,:].values
tmp_y_train = y_train[rnd_indx]

rnd_indx_2 = random.sample(range(df_train.shape[0]), 1000)
tmp_X_test = df_train.iloc[rnd_indx_2,:].values
tmp_y_test = y_train[rnd_indx_2]

# Following the tutorial on their site:
lr_model = xl.LRModel(task='reg', epoch=10, lr=0.1)
lr_model.fit(tmp_X_train, tmp_y_train)

# Ok, all NaN, I have tried a few parameter combinations and nothing works, so
# let's see the native methods. To use native methods one has to save the
# files to disk in svmlight form. Fortunately, sklearn has a funcitonality for
# that.
dump_svmlight_file(tmp_X_train, tmp_y_train, "trainfm.txt")

lr_model2 = xl.create_linear()
lr_model2.setTrain("trainfm.txt")
param = {'task':'reg', 'lr':0.1, 'epoch': 10}
lr_model2.fit(param, "model.out")

# OK, so at this point I thought I will use these methods and optimize + cross
# validate, given that the library comes with a convenient .cv method
lr_model2 = xl.create_linear()
lr_model2.setTrain("trainfm.txt")
param = {'task':'reg', 'lr':0.1, 'epoch': 10}
lr_model2.fit(param, "model.out")
lr_model2.cv(param)

model_methods = [method for method in dir(lr_model2)
 if callable(getattr(lr_model2, method))]

# so the output is beautiful :) but there is no way of accessing to the score!
# At this stage it became personal, so I will do all manually. Stay with me
# because it is going to be painful:

# 1-. I will take X% of the training data and split it in 3 folds
# 2-. Save them to disk in respective files
# 3-. perform cv manually within an hyperopt function
XLEARN_DIR = "xlearn_data"
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

def xl_objective(params, method="linear"):

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

# validation activities
df_purchases_valid = pd.read_pickle(os.path.join(inp_dir, 'valid', 'df_purchases_valid.p'))
df_visits_valid = pd.read_pickle(os.path.join(inp_dir, 'valid', 'df_visits_valid.p'))
df_visits_valid.rename(index=str, columns={'view_coupon_id_hash': 'coupon_id_hash'}, inplace=True)

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

valid_coupon_ids = df_coupons_valid_oh_feat.coupon_id_hash.values
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
df_valid = pd.merge(df_valid, df_users_train_oh_feat, on='user_id_hash')
df_valid = pd.merge(df_valid, df_coupons_valid_oh_feat, on = 'coupon_id_hash')

X_train = csr_matrix(df_train.values)
train_data_file = os.path.join(XLEARN_DIR,"xltrain.txt")
%time dump_svmlight_file(X_train,y_train,train_data_file)
del(X_train)

X_valid = csr_matrix(df_valid
    .drop(['user_id_hash','coupon_id_hash'], axis=1)
    .values)
y_valid = np.array([0.1]*X_valid.shape[0])
valid_data_file = os.path.join(XLEARN_DIR,"xlvalid.txt")
%time dump_svmlight_file(X_valid,y_valid,valid_data_file)
del(X_valid)

best_linear = pickle.load(open(os.path.join(XLEARN_DIR,'best_fm.p'), "rb"))
best_linear['epoch'] = int(best_linear['epoch'])
best_linear['k'] = int(best_linear['k'])
best_linear['task'] = 'reg'
best_linear['metric'] = 'rmse'

xlmodel_fname = os.path.join(XLEARN_DIR,"xllinear_model.out")
xlpreds_fname = os.path.join(XLEARN_DIR,"xllinear_preds.txt")

xl_model = xl.create_fm()
xl_model.setTrain(train_data_file)
xl_model.setTest(valid_data_file)

xl_model.fit(best_linear, xlmodel_fname)
xl_model.predict(xlmodel_fname, xlpreds_fname)

preds = np.loadtxt(xlpreds_fname)
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