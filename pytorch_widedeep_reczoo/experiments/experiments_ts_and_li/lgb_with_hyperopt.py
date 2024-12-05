import json
import warnings
from typing import Any, Dict, Literal
from pathlib import Path

import lightgbm as lgb
from hyperopt import Trials, hp, tpe, fmin, space_eval
from lightgbm import Dataset as lgbDataset
from sklearn.metrics import f1_score, accuracy_score, root_mean_squared_error
from pytorch_widedeep.utils import LabelEncoder

from rec_tools.constants import RESULTS_DIR
from rec_tools.prepare_experiments.prepare_ts_or_li import (
    experiment_with_feat_engineering,
)

warnings.filterwarnings("ignore")


class LGBOptimizerHyperopt(object):
    def __init__(
        self,
        verbose: bool = False,
        binary_target: bool = True,
        score: Literal["accuracy", "f1", "binary_logloss", "l2"] = "binary_logloss",
    ):
        self.verbose = verbose
        self.score = score
        self.binary_target = binary_target

        assert self.binary_target == (
            self.score in ["accuracy", "f1", "binary_logloss"]
        ), "binary_target must be True if score is accuracy, f1, or binary_logloss"

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
        best["objective"] = "binary" if self.binary_target else "regression"

        self.best.update(best)

    def get_objective(self, dtrain: lgbDataset, deval: lgbDataset):
        def objective(params: Dict[str, Any]) -> float:

            # hyperopt casts as float
            params["n_estimators"] = 1000
            params["verbose"] = -1
            params["seed"] = 1
            params["feature_pre_filter"] = False
            params["objective"] = "binary" if self.binary_target else "regression"

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
                score = (
                    model.best_score["valid_0"]["binary_logloss"]
                    if self.binary_target
                    else model.best_score["valid_0"]["l2"]
                )

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
    use_umap: Literal["st", "ch"],
    split_type: Literal["ts", "li"] = "ts",
    binary_target: bool = True,
) -> None:
    train_df, val_df, cat_cols = experiment_with_feat_engineering(
        use_umap, split_type, binary_target
    )

    encoder = LabelEncoder(columns_to_encode=cat_cols)

    train_df_encoded = encoder.fit_transform(train_df)
    val_df_encoded = encoder.transform(val_df)

    y_train = train_df_encoded["rating"]
    y_val = val_df_encoded["rating"]
    X_train = train_df_encoded.drop("rating", axis=1)
    X_val = val_df_encoded.drop("rating", axis=1)

    results_dir = (
        Path(RESULTS_DIR)
        / f"results_lgb_with_hyperopt_{use_umap}_{split_type}_{'binary' if binary_target else 'regression'}"
    )
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

    tuner = LGBOptimizerHyperopt(
        verbose=True,
        binary_target=binary_target,
        score="binary_logloss" if binary_target else "l2",
    )
    tuner.optimize(lgbtrain, lgbvalid)

    model = lgb.train(
        tuner.best,
        lgbtrain,
        valid_sets=[lgbvalid],
        callbacks=[lgb.early_stopping(50, verbose=True)],
    )

    tuner.best["n_estimators"] = model.best_iteration  # type: ignore

    _y_pred = model.predict(X_val)

    if binary_target:
        y_pred = (_y_pred > 0.5).astype(int)  # type: ignore
        accuracy = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        best_trial = {
            "best_params": tuner.best,
            "accuracy": accuracy,
            "f1": f1,
            "val_loss": model.best_score["valid_0"]["binary_logloss"],
        }
        print("Accuracy: ", accuracy)
        print("F1: ", f1)
    else:
        y_pred = _y_pred
        rmse = root_mean_squared_error(y_val, y_pred)
        best_trial = {
            "best_params": tuner.best,
            "rmse": rmse,
            "val_loss": model.best_score["valid_0"]["l2"],
        }
        print("RMSE: ", rmse)

    with open(results_dir / "results.json", "w") as f:
        json.dump(best_trial, f, indent=4)


if __name__ == "__main__":
    run_ts_lgb_with_hyperopt(use_umap="ch", split_type="ts", binary_target=True)
    run_ts_lgb_with_hyperopt(use_umap="ch", split_type="li", binary_target=True)
    run_ts_lgb_with_hyperopt(use_umap="ch", split_type="ts", binary_target=False)
    run_ts_lgb_with_hyperopt(use_umap="ch", split_type="li", binary_target=False)
