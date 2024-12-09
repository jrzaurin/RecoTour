from typing import cast

from rec_tools.process_experiments.utils import (
    SplitType,
    AllMetricType,
    get_best_experiment,
)
from rec_tools.process_experiments.process_results import process_experiment_results

if __name__ == "__main__":
    binary_metrics_dict = process_experiment_results("binary")

    for split_type in ["ts", "li"]:
        split_type_ = cast(SplitType, split_type)
        for metric in ["acc", "f1", "val_loss"]:
            metric_type = cast(AllMetricType, metric)
            best_exp_name, best_value = get_best_experiment(
                binary_metrics_dict, metric_type, split_type_
            )
            print(
                f"Best experiment by {metric} and {split_type}: {best_exp_name} ({metric}: {best_value:.3f})"
            )

    regression_metrics_dict = process_experiment_results("regression")

    for split_type in ["ts", "li"]:
        split_type_ = cast(SplitType, split_type)
        metric_type = cast(AllMetricType, "rmse")
        best_exp_name, best_value = get_best_experiment(
            regression_metrics_dict, metric_type, split_type_
        )
        print(
            f"Best experiment by rmse and {split_type}: {best_exp_name} (rmse: {best_value:.3f})"
        )
