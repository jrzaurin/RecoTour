import json
import pickle
from typing import Any, Dict, Tuple, Literal
from pathlib import Path


def find_results_json(directory):
    for path in directory.rglob("results.json"):
        return path
    return None


def load_experiment_results(
    results_dir: Path, experiment_type: Literal["binary", "regression"]
) -> Dict[str, Any]:
    results = {}
    exp_dirs = [
        d for d in results_dir.iterdir() if d.is_dir() and experiment_type in d.name
    ]

    for dir_path in exp_dirs:
        results_path = find_results_json(dir_path)

        if results_path:
            try:
                with open(results_path) as rp:
                    results[dir_path.name] = json.load(rp)
            except json.JSONDecodeError:
                print(f"Error reading JSON from {dir_path.name}")
        else:
            pickle_path = dir_path / "results.pkl"
            if pickle_path.exists():
                try:
                    with open(pickle_path, "rb") as pp:
                        results[dir_path.name] = pickle.load(pp)
                except (pickle.UnpicklingError, EOFError) as e:
                    raise RuntimeError(
                        f"Failed to load both JSON and pickle files in {dir_path.name}: {str(e)}"
                    )
            else:
                raise FileNotFoundError(
                    f"No results.json or results.pkl found in {dir_path.name}"
                )

    return results


def extract_binary_metrics(
    experiment_results: Dict[str, Any]
) -> Dict[str, Dict[str, float]]:
    metrics_dict = {}
    for exp_name, results in experiment_results.items():
        if "with_feature_elimination" in exp_name:
            acc, f1, val_loss = extract_exp_with_feature_elimination_best_trial_results(
                results_dict=results,
            )
        elif "ray" in exp_name:
            acc = results["metrics"]["accuracy"]
            f1 = results["metrics"]["f1"]
            val_loss = results["metrics"]["val_loss"]
        else:
            try:
                acc = results["acc"] if "acc" in results else results["accuracy"]
                f1 = results["f1"]
                val_loss = results["val_loss"]
            except KeyError:
                print(f"KeyError for {exp_name}")
                continue

        metrics_dict[exp_name] = {"acc": acc, "f1": f1, "val_loss": val_loss}

    return metrics_dict


def extract_regression_metrics(
    experiment_results: Dict[str, Any]
) -> Dict[str, Dict[str, float]]:
    metrics_dict = {}
    for exp_name, results in experiment_results.items():
        if "with_feature_elimination" in exp_name:
            rmse, _, val_loss = extract_exp_with_feature_elimination_best_trial_results(
                results_dict=results,
            )
        elif "ray" in exp_name:
            rmse = results["metrics"]["rmse"]
            val_loss = results["metrics"]["val_loss"]
        elif "svd" in exp_name:
            rmse = results["rmse"]
            val_loss = results["rmse"]
        else:
            try:
                rmse = results["rmse"]
                val_loss = results["val_loss"]
            except KeyError:
                print(f"KeyError for {exp_name}")
                continue

        metrics_dict[exp_name] = {"rmse": rmse, "val_loss": val_loss}

    return metrics_dict


def extract_exp_with_feature_elimination_best_trial_results(
    results_dict: Dict[int, Dict[str, Any]],
) -> Tuple[float, float | None, float]:
    best_trial = max(results_dict, key=lambda x: -results_dict[x]["val_loss"])
    if "acc" in results_dict[best_trial]:
        metric1 = results_dict[best_trial]["acc"]
        metric2 = results_dict[best_trial]["f1"]
    else:
        metric1 = results_dict[best_trial]["rmse"]
        metric2 = None
    val_loss = results_dict[best_trial]["val_loss"]
    return metric1, metric2, val_loss


AllMetricType = Literal["acc", "f1", "rmse", "val_loss"]

SplitType = Literal["ts", "li"]


def get_best_experiment(
    metrics_dict: dict, metric: AllMetricType, split_type: SplitType
) -> tuple[str, float]:
    _split_type = f"_{split_type}_"
    split_type_metrics_dict = {
        k: v for k, v in metrics_dict.items() if _split_type in k
    }
    reverse = metric != "val_loss" and metric != "rmse"
    best_exp = max(
        split_type_metrics_dict.items(),
        key=lambda x: x[1][metric] if reverse else -x[1][metric],
    )
    return best_exp[0], best_exp[1][metric]
