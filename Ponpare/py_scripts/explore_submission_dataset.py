import pandas as pd
import os


DATA_DIR = "../datasets/Ponpare/data_translated/"

df_coupon_list_test = pd.read_csv(os.path.join(DATA_DIR, "coupon_list_test.csv"))
df_coupon_list_train = pd.read_csv(os.path.join(DATA_DIR, "coupon_list_train.csv"))

df_l = [df_coupon_list_test, df_coupon_list_train]
for df in df_l:
	df['dispfrom'] = pd.to_datetime(df.dispfrom, infer_datetime_format=True)
	df['dispend'] = pd.to_datetime(df.dispend, infer_datetime_format=True)
	df['validfrom'] = pd.to_datetime(df.validfrom, infer_datetime_format=True)
	df['validend'] = pd.to_datetime(df.validend, infer_datetime_format=True)

print(df_coupon_list_train.dispfrom.max(), df_coupon_list_test.dispfrom.min())
print(df_coupon_list_train.validfrom.max(), df_coupon_list_test.validfrom.min())
print(df_coupon_list_test.dispfrom.min(), df_coupon_list_test.dispfrom.max())
print(df_coupon_list_test.dispfrom.min(), df_coupon_list_test.dispend.max())
print(df_coupon_list_train.dispperiod.mean(), df_coupon_list_train.dispperiod.median())


