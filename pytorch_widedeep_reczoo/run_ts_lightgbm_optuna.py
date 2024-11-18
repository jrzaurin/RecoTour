import pickle
import warnings
from typing import Any, Dict, Literal
from pathlib import Path

import lightgbm as lgb
from lightgbm import Dataset as lgbDataset
from sklearn.metrics import f1_score, accuracy_score
from optuna.integration import lightgbm

from rec_zoo.prepare_experiments.prepare_ts import prepare_experiment

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


def run_ts_lightgbm_optuna(
    use_umap: Literal["st", "ch"] = "st",
    save_dir: str = "results_lgbm_optuna",
    save_name: str = "optuna_results.pkl",
):
    train_df, val_df, _, encoder = prepare_experiment(use_umap=use_umap, gbm="lgbm")
    y_train = train_df["rating"]
    y_val = val_df["rating"]
    X_train = train_df.drop("rating", axis=1)
    X_val = val_df.drop("rating", axis=1)

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

    lgb_optimizer = LGBOptunaOptimizer()
    lgb_optimizer.optimize(lgbtrain, lgbvalid)

    model = lgb.train(
        lgb_optimizer.best,
        lgbtrain,
        valid_sets=[lgbvalid],
        callbacks=[lgb.early_stopping(50, verbose=True)],
    )

    y_pred = model.predict(X_val)
    y_pred_labels = (y_pred > 0.5).astype(int)  # type: ignore

    save_path = Path("results") / save_dir / save_name
    save_path.parent.mkdir(exist_ok=True, parents=True)

    best_trial = {
        "best_params": lgb_optimizer.best,
        "accuracy": accuracy_score(y_val, y_pred_labels),
        "f1": f1_score(y_val, y_pred_labels),
        "val_loss": model.best_score["valid_0"]["binary_logloss"],
    }
    with open(save_path, "wb") as bt:
        pickle.dump(best_trial, bt)

    print("Accuracy: ", accuracy_score(y_val, y_pred_labels))
    print("F1: ", f1_score(y_val, y_pred_labels))


if __name__ == "__main__":

    run_ts_lightgbm_optuna()
