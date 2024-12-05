import json
import pickle
import warnings
from typing import Any, Dict, Tuple, Literal
from pathlib import Path

import numpy as np
import pandas as pd
from hyperopt import Trials, hp, tpe, fmin, space_eval
from surprise import SVD, Reader, Dataset
from sklearn.metrics import f1_score, log_loss, accuracy_score, root_mean_squared_error

from rec_tools.constants import RESULTS_DIR
from rec_tools.prepare_experiments.prepare_ts_or_li import (
    experiment_without_feat_engineering,
)

warnings.filterwarnings("ignore")


class SVDOptimizerHyperopt:
    def __init__(
        self,
        verbose: bool = False,
        binary_target: bool = True,
        score: Literal["accuracy", "f1", "logloss", "rmse"] = "logloss",
    ):
        self.verbose = verbose
        self.score = score
        self.binary_target = binary_target

        assert self.binary_target == (
            self.score in ["accuracy", "f1", "logloss"]
        ), "binary_target must be True if score is accuracy, f1, or logloss"

        self.best: Dict[str, Any] = {}

    def optimize(
        self,
        train_data,
        val_df,
        maxevals: int = 200,
    ):
        param_space = self.hyperparameter_space()
        objective = self.get_objective(train_data, val_df)
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
        best["n_epochs"] = int(best["n_epochs"])
        best["n_factors"] = int(best["n_factors"])
        self.best.update(best)

    def get_objective(self, train_data, val_df):
        def objective(params: Dict[str, Any]) -> float:
            params["n_epochs"] = int(params["n_epochs"])
            params["n_factors"] = int(params["n_factors"])

            model = SVD(
                n_factors=params["n_factors"],
                n_epochs=params["n_epochs"],
                lr_all=params["lr_all"],
                reg_all=params["reg_all"],
            )
            model.fit(train_data)

            val_predictions = [
                model.predict(uid, iid).est
                for uid, iid in zip(val_df["user_id"], val_df["item_id"])
            ]

            if self.score == "logloss":
                score = log_loss(val_df["rating"], val_predictions)
            elif self.score == "accuracy":
                val_pred_labels = (np.array(val_predictions) > 0.5).astype(int)
                score = -accuracy_score(val_df["rating"], val_pred_labels)
            elif self.score == "f1":
                val_pred_labels = (np.array(val_predictions) > 0.5).astype(int)
                score = -f1_score(val_df["rating"], val_pred_labels)
            else:  # rmse
                score = root_mean_squared_error(val_df["rating"], val_predictions)

            return score

        return objective

    def hyperparameter_space(self) -> Dict[str, Any]:
        space = {
            "n_factors": hp.quniform("n_factors", 20, 200, 10),
            "n_epochs": hp.quniform("n_epochs", 10, 100, 5),
            "lr_all": hp.loguniform("lr_all", np.log(0.001), np.log(0.1)),
            "reg_all": hp.loguniform("reg_all", np.log(0.001), np.log(0.1)),
        }
        return space


def train_svd_with_hyperopt(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    binary_target: bool,
) -> Tuple[SVD, Dict[str, float]]:
    reader = Reader(rating_scale=(0, 1) if binary_target else (1, 5))
    train_data = Dataset.load_from_df(
        train_df[["user_id", "item_id", "rating"]], reader
    ).build_full_trainset()

    tuner = SVDOptimizerHyperopt(
        verbose=True,
        binary_target=binary_target,
        score="logloss" if binary_target else "rmse",
    )
    tuner.optimize(train_data, val_df)

    model = SVD(**tuner.best)
    model.fit(train_data)

    val_predictions = [
        model.predict(uid, iid).est
        for uid, iid in zip(val_df["user_id"], val_df["item_id"])
    ]

    if binary_target:
        val_pred_labels = (np.array(val_predictions) > 0.5).astype(int)
        acc = accuracy_score(val_df["rating"], val_pred_labels)
        f1 = f1_score(val_df["rating"], val_pred_labels)
        metrics = {
            "best_params": tuner.best,
            "accuracy": acc,
            "f1": f1,
        }
        print(f"SVD Accuracy: {acc:.4f}")
        print(f"SVD F1: {f1:.4f}")
    else:
        rmse = root_mean_squared_error(val_df["rating"], val_predictions)
        metrics = {
            "best_params": tuner.best,
            "rmse": rmse,
        }
        print(f"SVD RMSE: {rmse:.4f}")

    return model, metrics


def main(split_type: Literal["ts", "li"] = "ts", binary_target: bool = True) -> None:
    train_df, val_df, _ = experiment_without_feat_engineering(split_type, binary_target)

    results_dir = (
        Path(RESULTS_DIR)
        / f"results_svd_with_hyperopt_{split_type}_{'binary' if binary_target else 'regression'}"
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    svd_model, svd_metrics = train_svd_with_hyperopt(train_df, val_df, binary_target)

    with open(results_dir / "model.pkl", "wb") as f:
        pickle.dump(svd_model, f)

    with open(results_dir / "results.json", "w") as f:
        json.dump(svd_metrics, f, indent=4)


if __name__ == "__main__":
    main(split_type="ts", binary_target=True)
    main(split_type="li", binary_target=True)
    main(split_type="ts", binary_target=False)
    main(split_type="li", binary_target=False)
