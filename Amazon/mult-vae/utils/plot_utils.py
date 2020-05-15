import pickle
from glob import glob
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set(color_codes=True)
sns.set(context="notebook", font_scale=1.0, font="serif")

results_dir = Path("../results")


def prepare_dataframe_for_plot(
    dataset: str, dl_frame: str
) -> Tuple[Dict, pd.DataFrame]:
    """
    dl_frame: one of 'pt' or 'mx'
    dataset: one of 'movilens' or 'amazon'
    """

    anneal_schedules = [
        "_".join([dl_frame, "anneal_schedule", dataset, str(i) + ".p"])
        for i in [100, 200, 300]
    ]

    anneal_schedule_100 = pickle.load(open(results_dir / anneal_schedules[0], "rb"))
    anneal_schedule_200 = pickle.load(open(results_dir / anneal_schedules[1], "rb"))
    anneal_schedule_300 = pickle.load(open(results_dir / anneal_schedules[2], "rb"))

    best_anneal_epoch_100 = np.where(
        list(anneal_schedule_100.values()) == max(list(anneal_schedule_100.values()))
    )[0][0]
    best_anneal_epoch_200 = np.where(
        list(anneal_schedule_200.values()) == max(list(anneal_schedule_200.values()))
    )[0][0]
    best_anneal_epoch_300 = np.where(
        list(anneal_schedule_300.values()) == max(list(anneal_schedule_300.values()))
    )[0][0]
    best_anneal_epochs = {}
    best_anneal_epochs[100] = best_anneal_epoch_100
    best_anneal_epochs[200] = best_anneal_epoch_200
    best_anneal_epochs[300] = best_anneal_epoch_300

    df_100 = pd.DataFrame(
        {
            "epochs": range(len(anneal_schedule_100)),
            "anneal": list(anneal_schedule_100.keys()),
            "NDCG@100": list(anneal_schedule_100.values()),
        }
    )
    df_100["architecture"] = "[100,300]"
    df_200 = pd.DataFrame(
        {
            "epochs": range(len(anneal_schedule_200)),
            "anneal": list(anneal_schedule_200.keys()),
            "NDCG@100": list(anneal_schedule_200.values()),
        }
    )
    df_200["architecture"] = "[200,600]"
    df_300 = pd.DataFrame(
        {
            "epochs": range(len(anneal_schedule_300)),
            "anneal": list(anneal_schedule_300.keys()),
            "NDCG@100": list(anneal_schedule_300.values()),
        }
    )
    df_300["architecture"] = "[300,900]"

    df_anneal_schedules = pd.concat([df_100, df_200, df_300])

    return best_anneal_epochs, df_anneal_schedules


def plot_anneal_schedule_dataset(dataset: str):
    def epoch2anneal(x):
        return x * 1 / 170

    def anneal2epoch(x):
        return x * 170

    pt_best_anneal_epochs, pt_df_anneal_schedules = prepare_dataframe_for_plot(
        dataset, "pt"
    )
    mx_best_anneal_epochs, mx_df_anneal_schedules = prepare_dataframe_for_plot(
        dataset, "mx"
    )

    plt.figure(figsize=(15, 4))

    plt.subplot(1, 2, 1)
    ax1 = sns.lineplot(
        pt_df_anneal_schedules.epochs,
        pt_df_anneal_schedules["NDCG@100"],
        hue=pt_df_anneal_schedules.architecture,
        style=pt_df_anneal_schedules.architecture,
        linewidth=2.0,
        legend="full",
    )
    ax1.set_title("Pytorch")
    ax1.axvline(pt_best_anneal_epochs[100], 0, 1, color="blue", linewidth=0.5)
    ax1.axvline(pt_best_anneal_epochs[200], 0, 1, color="orange", linewidth=0.5)
    ax1.axvline(pt_best_anneal_epochs[300], 0, 1, color="green", linewidth=0.5)
    if dataset == "movielens":
        ax1.legend(bbox_to_anchor=(0.075, 0.39))
    ax2 = ax1.secondary_xaxis("top", functions=(epoch2anneal, anneal2epoch))

    plt.suptitle(dataset, fontweight="bold")
    plt.subplots_adjust(top=0.8)

    plt.subplot(1, 2, 2)
    ax1 = sns.lineplot(
        mx_df_anneal_schedules.epochs,
        mx_df_anneal_schedules["NDCG@100"],
        hue=mx_df_anneal_schedules.architecture,
        style=mx_df_anneal_schedules.architecture,
        linewidth=2.0,
        legend="full",
    )
    ax1.set_title("Mxnet")
    ax1.axvline(mx_best_anneal_epochs[100], 0, 1, color="blue", linewidth=0.5)
    ax1.axvline(mx_best_anneal_epochs[200], 0, 1, color="orange", linewidth=0.5)
    ax1.axvline(mx_best_anneal_epochs[300], 0, 1, color="green", linewidth=0.5)
    ax2 = ax1.secondary_xaxis("top", functions=(epoch2anneal, anneal2epoch))


def plot_anneal_schedule():
    plot_anneal_schedule_dataset("movielens")
    plot_anneal_schedule_dataset("amazon")


def find_best(dl_frame: str, model: str) -> pd.DataFrame:

    keep_cols = [
        "dataset",
        "dl_frame",
        "model",
        "p_dims",
        "weight_decay",
        "lr",
        "anneal_cap",
        "best_epoch",
        "loss",
        "n100",
        "r20",
        "r50",
    ]

    pattern = "_".join([dl_frame, model, "*"])
    search_dir = results_dir / pattern

    model_files = glob(str(search_dir))

    run_results = []
    for f in model_files:
        run_results.append(pickle.load(open(f, "rb")))

    sub_keys = [k for k in run_results[0].keys() if k != "args"]

    run_results_dfs = []
    for r in run_results:
        dict1 = r["args"]
        dict2 = {k: v for k, v in r.items() if k in sub_keys}
        df1 = pd.DataFrame(dict1, index=[0])
        df2 = pd.DataFrame(dict2, index=[0])
        run_results_dfs.append(pd.concat([df1, df2], axis=1))

    results_df = pd.concat(run_results_dfs)
    results_df = (
        results_df.sort_values(["dataset", "n100"], ascending=False)
        .groupby("dataset")
        .head(1)
        .reset_index(drop=True)
    )
    results_df["dl_frame"] = "Pytorch" if dl_frame == "pt" else "Mxnet"
    results_df.loc[results_df.model=='dae', 'anneal_cap'] = "NA"
    return results_df[keep_cols]


def plot_dataset_loss(df: pd.DataFrame, dataset: str):
    df_pt = df[(df.dataset == dataset) & (df.dl_frame == "Pytorch")][
        ["loss", "n100", "r20", "r50"]
    ].melt("loss", var_name="metric", value_name="value")
    df_mx = df[(df.dataset == dataset) & (df.dl_frame == "Mxnet")][
        ["loss", "n100", "r20", "r50"]
    ].melt("loss", var_name="metric", value_name="value")

    plt.figure(figsize=(15, 4))

    plt.subplot(1, 2, 1)
    sns.lineplot(
        x="loss",
        y="value",
        hue="metric",
        style="metric",
        markers=True,
        markersize=10,
        linewidth=2,
        data=df_pt,
        legend="full",
        palette="Reds",
    )
    plt.xlabel("Loss")
    plt.ylabel("metric value")
    plt.title("Pytorch")

    plt.subplot(1, 2, 2)
    sns.lineplot(
        x="loss",
        y="value",
        hue="metric",
        style="metric",
        markers=True,
        markersize=10,
        linewidth=2,
        data=df_mx,
        legend="full",
        palette="Blues",
    )
    plt.xlabel("Loss")
    plt.ylabel("metric value")
    plt.title("Mxnet")

    plt.suptitle(dataset, fontweight="bold")
    plt.subplots_adjust(top=0.8)


def plot_loss(df: pd.DataFrame):
    plot_dataset_loss(df, "movielens")
    plot_dataset_loss(df, "amazon")
