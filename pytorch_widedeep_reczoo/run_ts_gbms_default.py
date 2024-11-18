import json
import pickle
from typing import List, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
import catboost as ctb
import lightgbm as lgb
from sklearn.metrics import f1_score, accuracy_score
from pytorch_widedeep.utils import LabelEncoder


def prepare_data(
    split_path: Path,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    List[str],
]:

    expriment_path = split_path / "ts_movielens_splits"
    train_df = pd.read_csv(expriment_path / "train.csv")
    val_df = pd.read_csv(expriment_path / "val.csv")

    for df in [train_df, val_df]:
        df.drop(["timestamp", "title"], axis=1, inplace=True)
        df["rating"] = (df["rating"] >= 4).astype(int)

    cat_cols = [
        "user_id",
        "item_id",
        "gender",
        "genres",
        "age",
        "occupation",
        "zipcode",
    ]

    encoder = LabelEncoder(cat_cols)
    train_dfe = encoder.fit_transform(train_df)
    val_dfe = encoder.transform(val_df)
    X_train = train_dfe[cat_cols]
    y_train = train_dfe["rating"]
    X_val = val_dfe[cat_cols]
    y_val = val_dfe["rating"]

    return (  # type: ignore
        train_df,
        val_df,
        X_train,
        X_val,
        y_train,
        y_val,
        cat_cols,
    )


def train_lightgbm(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: pd.DataFrame,
    y_val: pd.DataFrame,
    cat_cols: List[str],
) -> Tuple[lgb.Booster, float, float]:
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
    val_pred_labels = (np.array(val_pred) > 0.5).astype(int)
    acc = accuracy_score(y_val, val_pred_labels)
    f1 = f1_score(y_val, val_pred_labels)
    print(f"LightGBM Accuracy: {acc:.4f}")
    print(f"LightGBM F1: {f1:.4f}")
    return model, acc, f1


def train_catboost(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    y_train: pd.DataFrame,
    y_val: pd.DataFrame,
    cat_cols: List[str],
) -> Tuple[ctb.CatBoost, float, float]:
    train_pool = ctb.Pool(train_df[cat_cols], label=y_train, cat_features=cat_cols)
    val_pool = ctb.Pool(val_df[cat_cols], label=y_val, cat_features=cat_cols)
    model = ctb.train(
        pool=train_pool,
        params={
            "loss_function": "Logloss",
            "eval_metric": "Accuracy",
            "early_stopping_rounds": 50,
        },
        eval_set=val_pool,
    )

    cat_val_pred = model.predict(val_pool, prediction_type="Probability")[:, 1]
    cat_val_pred_labels = (cat_val_pred > 0.5).astype(int)
    acc = accuracy_score(y_val, cat_val_pred_labels)
    f1 = f1_score(y_val, cat_val_pred_labels)
    print(f"CatBoost Accuracy: {acc:.4f}")
    print(f"CatBoost F1: {f1:.4f}")
    return model, acc, f1


def main() -> None:
    split_path = Path("train_val_test_splits")
    train_df, val_df, X_train, X_val, y_train, y_val, cat_cols = prepare_data(
        split_path
    )

    results_dir = Path("results/results_gbms_default")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Train models
    lgb_model, lgb_acc, lgb_f1 = train_lightgbm(
        X_train, X_val, y_train, y_val, cat_cols
    )
    ctb_model, ctb_acc, ctb_f1 = train_catboost(
        train_df, val_df, y_train, y_val, cat_cols
    )

    with open(results_dir / "lightgbm_model.pkl", "wb") as f:
        pickle.dump(lgb_model, f)

    with open(results_dir / "catboost_model.pkl", "wb") as f:
        pickle.dump(ctb_model, f)

    # Save metrics
    metrics = {
        "lightgbm": {"accuracy": lgb_acc, "f1": lgb_f1},
        "catboost": {"accuracy": ctb_acc, "f1": ctb_f1},
    }

    with open(results_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    main()
