import numpy as np
import pandas as pd
import os

# This approach will be perhaps the one easier to understand. We have user
# features, item features and a target (interest), so let's turn this into a
# supervised problem and fit a regressor. Since this is a "standard" technique
# I will use this opportunity to illustrate a variety of tools around ML in
# general and boosted methods in particular

inp_dir = "../datasets/Ponpare/data_processed/"
train_dir = "train"
valid_dir = "valid"

# train and validation coupon features
df_coupons_train_feat = pd.read_pickle(os.path.join(inp_dir, train_dir, 'df_coupons_train_feat.p'))
df_coupons_valid_feat = pd.read_pickle(os.path.join(inp_dir, valid_dir, 'df_coupons_valid_feat.p'))

# In general, when using boosted methods, the presence of correlated or
# redundant features is not a big deal, since these will be ignored through
# the boosting rounds. However, for clarity and to reduce the chances of
# overfitting, we will select a subset of features here. All the numerical
# features have corresponding categorical ones, so we will keep those in
# moving forward. In addition, if we remember, for valid period, validend and
# validfrom, we used to methods to inpute NaN. Method1: Considering NaN as
# another category and Method2: replace NaN first in the object/numeric column
# and then turning the column into categorical. To start with, we will use
# Method1 here.
drop_cols = [c for c in df_coupons_train_feat.columns if ('_cat' not in c) or ('method2' in c)]
df_coupons_train_cat_feat = df_coupons_train_feat.drop(drop_cols, axis=1)
df_coupons_valid_cat_feat = df_coupons_valid_feat.drop(drop_cols, axis=1)

# train user features: there are a lot of features for users, both, numerical
# and categorical. We keep them all
df_users_train_feat = pd.read_pickle(os.path.join(inp_dir, train_dir, 'df_user_train_feat.p'))

# interest dataframe
df_interest = pd.read_pickle(os.path.join(inp_dir, train_dir, 'df_interest.p'))

df_train =