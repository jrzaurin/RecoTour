import pickle
from typing import Any, Dict, List, Tuple, Literal
from pathlib import Path

import pandas as pd
from catboost import Pool, CatBoostRegressor, CatBoostClassifier
from sklearn.metrics import f1_score, accuracy_score, root_mean_squared_error

from rec_tools.constants import RESULTS_DIR
from rec_tools.prepare_experiments.prepare_ts_or_li import (
    experiment_with_feat_engineering,
)


def create_initial_datasets(
    use_umap: Literal["st", "ch"],
    split_type: Literal["ts", "li"],
    binary_target: bool,
) -> Tuple[Pool, Pool, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str]]:
    train_df, val_df, cat_cols = experiment_with_feat_engineering(
        use_umap, split_type, binary_target
    )

    y_train = train_df["rating"]
    y_val = val_df["rating"]
    X_train = train_df.drop("rating", axis=1)
    X_val = val_df.drop("rating", axis=1)

    train_data = Pool(X_train, label=y_train, cat_features=cat_cols)
    val_data = Pool(X_val, label=y_val, cat_features=cat_cols)

    return train_data, val_data, X_train, X_val, y_train, y_val, cat_cols


def run_catboost_native_feature_selection(
    use_umap: Literal["st", "ch"],
    split_type: Literal["ts", "li"],
    binary_target: bool,
    select_features_algorithm: Literal[
        "RecursiveByPredictionValuesChange",
        "RecursiveByLossFunctionChange",
        "RecursiveByShapValues",
    ] = "RecursiveByLossFunctionChange",
) -> Dict[str, Any]:
    train_data, val_data, X_train, X_val, y_train, y_val, cat_cols = (
        create_initial_datasets(use_umap, split_type, binary_target)
    )

    select_features_algorithm_suffix_map = {
        "RecursiveByPredictionValuesChange": "pvc",
        "RecursiveByLossFunctionChange": "lfc",
        "RecursiveByShapValues": "shap",
    }

    if binary_target:
        model = CatBoostClassifier(
            num_boost_round=500,
            loss_function="Logloss",
            eval_metric="Logloss",
            early_stopping_rounds=50,
            verbose=True,
            allow_writing_files=False,
        )
    else:
        model = CatBoostRegressor(
            num_boost_round=500,
            loss_function="RMSE",
            eval_metric="RMSE",
            early_stopping_rounds=50,
            verbose=True,
            allow_writing_files=False,
        )

    summary = model.select_features(
        train_data,
        eval_set=val_data,
        num_features_to_select=10,
        features_for_select=train_data.get_feature_names(),
        algorithm=select_features_algorithm,
        logging_level="Silent",
    )

    selected_features = summary["selected_features_names"]

    model.fit(
        X_train[selected_features],
        y_train,
        cat_features=[c for c in cat_cols if c in selected_features],
        eval_set=(X_val[selected_features], y_val),
        verbose=True,
    )

    if binary_target:
        y_pred = model.predict_proba(X_val[selected_features])[:, 1]
        y_pred_labels = (y_pred > 0.5).astype(int)
        results = {
            "features": selected_features,
            "accuracy": accuracy_score(y_val, y_pred_labels),
            "f1": f1_score(y_val, y_pred_labels),
            "val_loss": model.get_best_score()["validation"]["Logloss"],
            "best_iteration": model.get_best_iteration(),
        }
    else:
        results = {
            "features": selected_features,
            "rmse": root_mean_squared_error(
                y_val, model.predict(X_val[selected_features])
            ),
            "val_loss": model.get_best_score()["validation"]["RMSE"],
            "best_iteration": model.get_best_iteration(),
        }

    print("-" * 100)
    if binary_target:
        print(
            f"Final metrics: accuracy: {results['accuracy']}, "
            f"f1: {results['f1']}, val_loss: {results['val_loss']}"
        )
    else:
        print(
            f"Final metrics: rmse: {results['rmse']}, val_loss: {results['val_loss']}"
        )

    print(f"Selected features: {selected_features}")
    print("-" * 100)

    sf_suffix = select_features_algorithm_suffix_map[select_features_algorithm]
    binary_target_suffix = "binary" if binary_target else "regression"
    results_dir = (
        Path(RESULTS_DIR)
        / f"results_ctb_with_native_feature_selection_{use_umap}_{split_type}_{sf_suffix}_{binary_target_suffix}"
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "results.pkl", "wb") as f:
        pickle.dump(results, f)

    return results


if __name__ == "__main__":
    # We will run only shap for now, otherwise we will stay here forever...
    results_ts_shap = run_catboost_native_feature_selection(
        use_umap="ch",
        split_type="ts",
        select_features_algorithm="RecursiveByShapValues",
        binary_target=True,
    )
    results_li_shap = run_catboost_native_feature_selection(
        use_umap="ch",
        split_type="li",
        select_features_algorithm="RecursiveByShapValues",
        binary_target=True,
    )
    results_ts_shap = run_catboost_native_feature_selection(
        use_umap="ch",
        split_type="ts",
        select_features_algorithm="RecursiveByShapValues",
        binary_target=False,
    )
    results_li_shap = run_catboost_native_feature_selection(
        use_umap="ch",
        split_type="li",
        select_features_algorithm="RecursiveByShapValues",
        binary_target=False,
    )
