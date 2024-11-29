import json
import warnings
from typing import Any, Dict, Literal
from pathlib import Path

import catboost as ctb
from hyperopt import Trials, hp, tpe, fmin, space_eval
from sklearn.metrics import f1_score, accuracy_score
from pytorch_widedeep.utils import LabelEncoder

from rec_tools.constants import RESULTS_DIR
from rec_tools.prepare_experiments.prepare_ts_or_li import (
    experiment_with_feature_engineering,
)

warnings.filterwarnings("ignore")


class CTBOptimizerHyperopt(object):
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
        dtrain: ctb.Pool,
        deval: ctb.Pool,
        maxevals: int = 100,
    ):
        # Initialize with default params
        self.best = {
            "iterations": 500,
            "early_stopping_rounds": 50,
            "verbose": False,
            "loss_function": "Logloss",
        }

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
        best["depth"] = int(best["depth"])
        best["min_data_in_leaf"] = int(best["min_data_in_leaf"])

        self.best.update(best)

    def get_objective(self, dtrain: ctb.Pool, deval: ctb.Pool):
        def objective(params: Dict[str, Any]) -> float:
            params["iterations"] = 500
            params["early_stopping_rounds"] = 50
            params["verbose"] = 1
            params["loss_function"] = "Logloss"
            params["depth"] = int(params["depth"])
            params["min_data_in_leaf"] = int(params["min_data_in_leaf"])

            model = ctb.train(
                pool=dtrain,
                params=params,
                eval_set=deval,
            )

            preds = model.predict(deval, prediction_type="Probability")[:, 1]
            if self.score == "accuracy":
                preds_labels = (preds > 0.5).astype(int)
                score = -accuracy_score(deval.get_label(), preds_labels)
            elif self.score == "f1":
                preds_labels = (preds > 0.5).astype(int)
                score = -f1_score(deval.get_label(), preds_labels)
            else:
                score = model.get_best_score()["validation"]["Logloss"]

            return score

        return objective

    def hyperparameter_space(
        self, param_space: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        space = {
            "learning_rate": hp.uniform("learning_rate", 0.01, 0.3),
            "depth": hp.quniform("depth", 4, 10, 1),
            "min_data_in_leaf": hp.quniform("min_data_in_leaf", 5, 50, 5),
            "l2_leaf_reg": hp.loguniform("l2_leaf_reg", -10, 0),  # exp(-10) to exp(0)
        }
        if param_space:
            return param_space
        else:
            return space


def run_ts_lgb_with_hyperopt(
    use_umap: Literal["st", "ch"], split_type: Literal["ts", "li"] = "ts"
) -> None:
    train_df, val_df, cat_cols = experiment_with_feature_engineering(
        use_umap, split_type
    )

    encoder = LabelEncoder(columns_to_encode=cat_cols)

    train_df_encoded = encoder.fit_transform(train_df)
    val_df_encoded = encoder.transform(val_df)

    y_train = train_df_encoded["rating"]
    y_val = val_df_encoded["rating"]
    X_train = train_df_encoded.drop("rating", axis=1)
    X_val = val_df_encoded.drop("rating", axis=1)

    results_dir = (
        Path(RESULTS_DIR) / f"results_ctb_with_hyperopt_{use_umap}_{split_type}"
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    train_pool = ctb.Pool(
        X_train,
        label=y_train,
        cat_features=encoder.columns_to_encode,
    )
    valid_pool = ctb.Pool(
        X_val,
        label=y_val,
        cat_features=encoder.columns_to_encode,
    )

    tuner = CTBOptimizerHyperopt(verbose=True)
    tuner.optimize(train_pool, valid_pool)

    model = ctb.train(
        pool=train_pool,
        params=tuner.best,
        eval_set=valid_pool,
    )

    y_pred = model.predict(X_val, prediction_type="Probability")[:, 1]
    y_pred_labels = (y_pred > 0.5).astype(int)  # type: ignore

    accuracy = accuracy_score(y_val, y_pred_labels)
    f1 = f1_score(y_val, y_pred_labels)

    best_trial = {
        "best_params": tuner.best,
        "accuracy": accuracy,
        "f1": f1,
        "val_loss": model.get_best_score()["validation"]["Logloss"],
    }

    with open(results_dir / "best_trial.json", "w") as f:
        json.dump(best_trial, f, indent=4)

    print("Accuracy: ", accuracy)
    print("F1: ", f1)


if __name__ == "__main__":
    # run_ts_lgb_with_hyperopt(use_umap="ch", split_type="ts")
    run_ts_lgb_with_hyperopt(use_umap="ch", split_type="li")
