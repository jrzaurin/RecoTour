import json
import warnings
from typing import Any, Dict, Literal
from pathlib import Path

import lightgbm as lgb
from lightgbm import Dataset as lgbDataset
from sklearn.metrics import f1_score, accuracy_score
from optuna.integration import lightgbm
from pytorch_widedeep.utils import LabelEncoder

from rec_tools.constants import RESULTS_DIR
from rec_tools.prepare_experiments.prepare_ts import experiment_with_feature_engineering

warnings.filterwarnings("ignore")


class LGBOptunaOptimizer(object):
    def __init__(
        self,
        verbose: bool = False,
    ):
        """
        Simple class that wraps up funcionality around LightGBMTuner
        """
        self.verbose = verbose
        self.best: Dict[str, Any] = {}

    def optimize(self, dtrain: lgbDataset, deval: lgbDataset):

        params: Dict[str, Any] = {"objective": "binary"}
        if self.verbose:
            params["verbosity"] = 1
        else:
            params["verbosity"] = -1

        params["early_stopping_rounds"] = 100

        self.tuner = lightgbm.LightGBMTuner(
            params=params,
            train_set=dtrain,
            valid_sets=[deval],
            num_boost_round=1000,
        )

        self.tuner.run()

        self.best = self.tuner.best_params
        # since n_estimators is not among the params that Optuna optimizes we
        # need to add it manually. We add a high value since it will be used
        # with early_stopping_rounds
        self.best["n_estimators"] = 1000  # type: ignore


def run_ts_lightgbm_optuna(use_umap: Literal["st", "ch"]) -> None:
    train_df, val_df, cat_cols = experiment_with_feature_engineering(use_umap)

    encoder = LabelEncoder(columns_to_encode=cat_cols)

    train_df_encoded = encoder.fit_transform(train_df)
    val_df_encoded = encoder.transform(val_df)

    y_train = train_df_encoded["rating"]
    y_val = val_df_encoded["rating"]
    X_train = train_df_encoded.drop("rating", axis=1)
    X_val = val_df_encoded.drop("rating", axis=1)

    results_dir = Path(RESULTS_DIR) / f"results_lgb_with_optuna_{use_umap}"
    results_dir.mkdir(parents=True, exist_ok=True)

    lgbtrain = lgbDataset(
        X_train,
        y_train,
        categorical_feature=encoder.columns_to_encode,
        free_raw_data=False,
    )
    lgbvalid = lgbDataset(
        X_val,
        y_val,
        reference=lgbtrain,
        free_raw_data=False,
    )

    tuner = LGBOptunaOptimizer()
    tuner.optimize(lgbtrain, lgbvalid)

    model = lgb.train(
        tuner.best,
        lgbtrain,
        valid_sets=[lgbvalid],
        callbacks=[lgb.early_stopping(50, verbose=True)],
    )

    y_pred = model.predict(X_val)
    y_pred_labels = (y_pred > 0.5).astype(int)  # type: ignore

    accuracy = accuracy_score(y_val, y_pred_labels)
    f1 = f1_score(y_val, y_pred_labels)

    best_trial = {
        "best_params": tuner.best,
        "accuracy": accuracy,
        "f1": f1,
        "val_loss": model.best_score["valid_0"]["binary_logloss"],
    }

    save_fname = results_dir / "optuna_results.pkl"
    with open(save_fname, "w") as f:
        json.dump(best_trial, f, indent=4)

    print("Accuracy: ", accuracy)
    print("F1: ", f1)


if __name__ == "__main__":

    run_ts_lightgbm_optuna(use_umap="ch")
