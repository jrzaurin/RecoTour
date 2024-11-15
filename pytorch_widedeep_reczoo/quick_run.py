from pathlib import Path

import pandas as pd
import catboost as ctb
import lightgbm as lgb
from sklearn.metrics import accuracy_score
from pytorch_widedeep.utils import LabelEncoder

split_path = Path("train_val_test_splits")
expriment_path = split_path / "ts_movielens_splits"

train_df = pd.read_csv(expriment_path / "train.csv")
val_df = pd.read_csv(expriment_path / "val.csv")

for df in [train_df, val_df]:
    df.drop(["timestamp", "title"], axis=1, inplace=True)
    df["rating"] = (df["rating"] >= 4).astype(int)

cat_cols = ["user_id", "item_id", "gender", "genres", "age", "occupation", "zipcode"]

encoder = LabelEncoder(cat_cols)
train_dfe = encoder.fit_transform(train_df)
val_dfe = encoder.transform(val_df)
X_train = train_dfe[cat_cols]
y_train = train_dfe["rating"]
X_val = val_dfe[cat_cols]
y_val = val_dfe["rating"]

# LightGBM
train_dataset = lgb.Dataset(
    X_train, label=y_train, categorical_feature=cat_cols, free_raw_data=False
)
val_dataset = lgb.Dataset(
    X_val, label=y_val, reference=train_dataset, free_raw_data=False
)
model = lgb.train(
    {
        "objective": "binary",
        "metric": "binary_logloss",
    },
    train_dataset,
    valid_sets=[train_dataset, val_dataset],
    valid_names=["train", "valid"],
    callbacks=[lgb.early_stopping(50, verbose=True), lgb.log_evaluation(period=1)],
)

val_pred = model.predict(X_val)
val_pred_labels = (val_pred > 0.5).astype(int)  # type: ignore[operator]
acc = accuracy_score(y_val, val_pred_labels)
print(f"LightGBM Accuracy: {acc:.4f}")

# CatBoost
train_pool = ctb.Pool(train_df[cat_cols], label=y_train, cat_features=cat_cols)
val_pool = ctb.Pool(val_df[cat_cols], label=y_val, cat_features=cat_cols)
catboost_model = ctb.train(
    pool=train_pool,
    params={
        "loss_function": "Logloss",
        "eval_metric": "Accuracy",
        "early_stopping_rounds": 50,
    },
    eval_set=val_pool,
)

# Make predictions and evaluate
cat_val_pred = catboost_model.predict(val_pool, prediction_type="Probability")[:, 1]
cat_val_pred_labels = (cat_val_pred > 0.5).astype(int)
cat_acc = accuracy_score(y_val, cat_val_pred_labels)
print(f"CatBoost Accuracy: {cat_acc:.4f}")
