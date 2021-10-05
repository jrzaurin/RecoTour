import pickle
from glob import glob
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set(color_codes=True)
sns.set(context="notebook", font_scale=1.0, font="serif")

results_dir = Path(
    "/Users/javier/ml_experiments_python/RecoTour/Amazon/mult-vae/results"
)


def prepare_dataframe_for_anneal_schedule_plot(
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

    (
        pt_best_anneal_epochs,
        pt_df_anneal_schedules,
    ) = prepare_dataframe_for_anneal_schedule_plot(dataset, "pt")
    (
        mx_best_anneal_epochs,
        mx_df_anneal_schedules,
    ) = prepare_dataframe_for_anneal_schedule_plot(dataset, "mx")

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
        "lr_scheduler",
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
    results_df.loc[results_df.model == "dae", "anneal_cap"] = "NA"

    return results_df[keep_cols].round(3)


def plot_dataset_loss(dataset: str):

    df = all_results_df()

    df_pt = df[(df.dataset == dataset) & (df.dl_frame == "Pytorch")][
        ["loss", "n100", "r20", "r50"]
    ].melt("loss", var_name="metric", value_name="value")
    df_mx = df[(df.dataset == dataset) & (df.dl_frame == "Mxnet")][
        ["loss", "n100", "r20", "r50"]
    ].melt("loss", var_name="metric", value_name="value")

    # manually removing an results that were so bad that mess the plot
    if dataset == "movielens":
        df_mx = df_mx[df_mx.loss < 380]
    elif dataset == "amazon":
        df_mx = df_mx[df_mx.loss < 95]
        df_pt = df_pt[df_pt.loss < 105]

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


def plot_metric_vs_loss():
    plot_dataset_loss("movielens")
    plot_dataset_loss("amazon")


def plot_ndcg_vs_pdims():
    def get_dae_experiments(dataset: str, keep_keys: List[str]):
        dataset_res = []
        for f, r in zip(model_files, results):
            if (
                (r["args"]["dataset"] == dataset)
                & (r["args"]["lr"] == 0.001)
                & (len(eval(r["args"]["p_dims"])) == 2)
                & (r["args"]["weight_decay"] == 0.0)
            ):
                r["args"]["dl_frame"] = f.split("/")[-1].split("_")[0]
                r["args"]["nunits"] = eval(r["args"]["p_dims"])[0]
                out_r = {k: v for k, v in r["args"].items() if k in keep_keys}
                out_r["n100"] = r["n100"]
                dataset_res.append(out_r)
        return dataset_res

    keep_keys = [
        "dataset",
        "dl_frame",
        "nunits",
        "n100",
    ]

    pattern = "*dae*"
    search_dir = results_dir / pattern

    model_files = glob(str(search_dir))

    results = []
    for f in model_files:
        results.append(pickle.load(open(f, "rb")))

    movielens_res = get_dae_experiments("movielens", keep_keys)
    amazon_res = get_dae_experiments("amazon", keep_keys)

    movielens_df = pd.concat(
        [pd.DataFrame(d, index=[0]) for d in movielens_res]
    ).reset_index(drop=True)
    amazon_df = pd.concat([pd.DataFrame(d, index=[0]) for d in amazon_res]).reset_index(
        drop=True
    )

    plt.figure(figsize=(15, 4))

    plt.subplot(1, 2, 1)
    sns.lineplot(
        x="nunits",
        y="n100",
        linewidth=2,
        data=movielens_df[movielens_df.dl_frame == "pt"],
        color="indianred",
        label="Pytorch",
        legend="full",
    )
    sns.lineplot(
        x="nunits",
        y="n100",
        linewidth=2,
        data=movielens_df[movielens_df.dl_frame == "mx"],
        label="Mxnet",
        legend="full",
    )
    plt.xlabel("p dims")
    plt.title("movielens")
    plt.xticks((50, 100, 200, 300))

    plt.subplot(1, 2, 2)
    sns.lineplot(
        x="nunits",
        y="n100",
        linewidth=2,
        data=amazon_df[amazon_df.dl_frame == "pt"],
        color="indianred",
        label="Pytorch",
        legend="full",
    )
    sns.lineplot(
        x="nunits",
        y="n100",
        linewidth=2,
        data=amazon_df[amazon_df.dl_frame == "mx"],
        label="Mxnet",
        legend="full",
    )
    plt.xlabel("p dims")
    plt.title("Amazon")
    plt.xticks((50, 100, 200, 300))


def build_results_df(dl_frame: str, model: str):

    keep_cols = {
        "dataset",
        "model",
        "dl_frame",
        "p_dims",
        "dropout_enc",
        "dropout_dec",
        "weight_decay",
        "lr",
        "batch_size",
        "anneal_cap",
        "lr_scheduler",
        "lr_patience",
        "early_stop_patience",
        "best_epoch",
        "loss",
        "n100",
        "r20",
        "r50",
    }

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
    results_df["dl_frame"] = "Pytorch" if dl_frame == "pt" else "Mxnet"
    results_df.loc[results_df.model == "dae", "anneal_cap"] = "NA"
    results_df = results_df[keep_cols].round(3)
    return results_df


def all_results_df():

    all_results = pd.concat(
        [
            build_results_df(dl_frame="pt", model="vae"),
            build_results_df(dl_frame="pt", model="dae"),
            build_results_df(dl_frame="mx", model="vae"),
            build_results_df(dl_frame="mx", model="dae"),
        ]
    ).reset_index(drop=True)

    return all_results
