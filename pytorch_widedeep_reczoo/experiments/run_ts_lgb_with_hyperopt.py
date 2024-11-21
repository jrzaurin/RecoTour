import json
import pickle
import warnings
from typing import Any, Dict, Literal
from pathlib import Path

import lightgbm as lgb
from hyperopt import Trials, hp, tpe, fmin, space_eval
from lightgbm import Dataset as lgbDataset
from sklearn.metrics import f1_score, accuracy_score

from rec_tools.constants import DATA_AND_ARTIFACTS_DIR
from rec_tools.prepare_experiments.prepare_ts import (
    prepare_experiment_with_feature_engineering,
)

warnings.filterwarnings("ignore")


class LGBOptimizerHyperopt(object):
    def __init__(
        self,
        verbose: bool = False,
        score: Literal["accuracy", "f1", "binary_logloss"] = "binary_logloss",
    ):
        self.verbose = verbose
        self.score = score
        self.max_or_min = "max" if score in ["accuracy", "f1"] else "min"

    def optimize(
        self,
        dtrain: lgbDataset,
        deval: lgbDataset,
        maxevals: int = 200,
    ):

        self.best = lgb.LGBMClassifier().get_params()

        param_space = self.hyperparameter_space()
        objective = self.get_objective(dtrain, deval)
        trials = Trials()
        best = fmin(
            fn=objective,
            space=param_space,
            algo=tpe.suggest,
            max_evals=maxevals,
            trials=trials,
            verbose=self.verbose,
        )
        self.trials = trials
        best = space_eval(param_space, trials.argmin)
        best["num_leaves"] = int(best["num_leaves"])
        best["min_child_samples"] = int(best["min_child_samples"])
        best["verbose"] = -1
        best["objective"] = "binary"

        # just a big number, since it will run with early stopping
        best["n_estimators"] = 1000

        self.best.update(best)

    def get_objective(self, dtrain: lgbDataset, deval: lgbDataset):
        def objective(params: Dict[str, Any]) -> float:

            # hyperopt casts as float
            params["n_estimators"] = 1000
            params["verbose"] = -1
            params["seed"] = 1
            params["feature_pre_filter"] = False
            params["objective"] = "binary"

            params["num_leaves"] = int(params["num_leaves"])
            params["min_child_samples"] = int(params["min_child_samples"])

            model = lgb.train(
                params,
                dtrain,
                valid_sets=[deval],
                callbacks=[lgb.early_stopping(50)],
            )

            preds = model.predict(deval.data)
            if self.score == "accuracy":
                preds_labels = (preds > 0.5).astype(int)  # type: ignore
                score = -accuracy_score(deval.label, preds_labels)
            elif self.score == "f1":
                preds_labels = (preds > 0.5).astype(int)  # type: ignore
                score = -f1_score(deval.label, preds_labels)
            else:
                score = model.best_score["valid_0"]["binary_logloss"]

            return score

        return objective

    def hyperparameter_space(
        self, param_space: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        space = {
            "learning_rate": hp.uniform("learning_rate", 0.01, 0.3),
            "num_leaves": hp.quniform("num_leaves", 20, 200, 10),
            "min_child_samples": hp.quniform("min_child_samples", 20, 100, 20),
            "colsample_bytree": hp.uniform("colsample_bytree", 0.5, 1.0),
            "reg_alpha": hp.choice(
                "reg_alpha", [0.01, 0.05, 0.1, 0.2, 0.4, 1.0, 2.0, 4.0, 10.0]
            ),
            "reg_lambda": hp.choice(
                "reg_lambda", [0.01, 0.05, 0.1, 0.2, 0.4, 1.0, 2.0, 4.0, 10.0]
            ),
        }
        if param_space:
            return param_space
        else:
            return space


def run_ts_lgb_with_hyperopt(
    use_umap: Literal["st", "ch"] = "st",
):
    train_df, val_df, _, encoder = prepare_experiment_with_feature_engineering(
        use_umap=use_umap, gbm="lgbm"
    )

    y_train = train_df["rating"]
    y_val = val_df["rating"]
    X_train = train_df.drop("rating", axis=1)
    X_val = val_df.drop("rating", axis=1)

    results_dir = Path(DATA_AND_ARTIFACTS_DIR) / "results/results_lgb_with_hyperopt"
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

    tuner = LGBOptimizerHyperopt(verbose=True)
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

    save_fname = results_dir / "hyperopt_results.pkl"
    with open(save_fname, "wb") as bt:
        pickle.dump(best_trial, bt)

    metrics = {
        "lgbm": {"accuracy": accuracy, "f1": f1},
    }

    with open(results_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("Accuracy: ", accuracy)
    print("F1: ", f1)


if __name__ == "__main__":
    run_ts_lgb_with_hyperopt(use_umap="st")
