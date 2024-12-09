import json
from typing import Literal
from pathlib import Path

import pandas as pd

from rec_tools.constants import RESULTS_DIR
from rec_tools.process_experiments.utils import (
    extract_binary_metrics,
    load_experiment_results,
    extract_regression_metrics,
)


def process_experiment_results(
    experiment_type: Literal["binary", "regression"]
) -> dict:

    results_path = Path(RESULTS_DIR)
    results = load_experiment_results(results_path, experiment_type)

    metrics_path = Path(RESULTS_DIR) / f"{experiment_type}_results"
    metrics_path.mkdir(parents=True, exist_ok=True)

    metrics_fname = metrics_path / f"{experiment_type}_metrics.json"
    metrics_dict = (
        extract_binary_metrics(results)
        if experiment_type == "binary"
        else extract_regression_metrics(results)
    )

    with open(metrics_fname, "w") as md:
        json.dump(metrics_dict, md, indent=2)

    results_df = pd.DataFrame(metrics_dict).transpose().reset_index(names="experiment")
    results_df.to_csv(
        metrics_fname.parent / f"{experiment_type}_metrics.csv", index=False
    )

    return metrics_dict
