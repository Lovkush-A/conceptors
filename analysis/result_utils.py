import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from glob import glob


def read_all_csvs(base_fname, return_list=False):
    fnames = glob(f"{base_fname}*.csv")
    dfs = [pd.read_csv(e) for e in fnames]
    if return_list:
        return dfs
    else:
        return pd.concat(dfs)


def get_baseline_accuracies(model_id: str, print_results: bool = False):
    tasks = ["antonyms", "capitalize", "country-capital", "english-french", "present-past", "singular-plural"]
    if print_results:
        print("Baseline accuracy for model:", model_id)
    for task in tasks:
        df = pd.read_csv(f"../results/{model_id}_{task}_baseline.csv")
        # NOTE: baseline_accuracy is in percent (not decimal)
        df = df[df.experiment == "Overall"]['baseline_accuracy'] / 100
        assert df.shape[0] == 1
        if print_results:
            print(f"{task:20} {df.values[0]:.2%}")


def plot_best_accuracy_per_layer_two_rows(model_id: str):
    tasks = ["antonyms", "capitalize", "country-capital", "english-french", "present-past", "singular-plural"]
    fig, axs = plt.subplots(2, 3, figsize=(12, 4), dpi=200, sharex=True, sharey=True)
    model_dict = {"gpt-j-6B": "GPT-J (6B)", "gpt-neox-20b": "GPT-NeoX (20B)"}
    fig.suptitle(f"Steering of {model_dict[model_id]}")
    for task_idx, task in enumerate(tasks):
        df = read_all_csvs(f"../results/{model_id}_{task}_addition")
        df = df[df.experiment == "Average"]
        df_max = df.groupby('layer').max().reset_index()
        ax = axs[task_idx // 3][task_idx % 3]
        ax.set_title(task)
        ax.plot(df_max.layer, df_max.final_accuracy, label='addition', lw=2)
        df = read_all_csvs(f"../results/{model_id}_{task}_conceptor")
        df = df[df.experiment == "Average"]
        df_max = df.groupby('layer').max().reset_index()
        ax.plot(df_max.layer, df_max.final_accuracy, label='conceptor', lw=2)
        ax.plot(df_max.layer, [0] * len(df_max.layer), label='baseline', lw=2, ls='--', color='red')
        if task_idx == 0:
            ax.legend()
        if task_idx % 3 == 0:
            ax.set_ylabel('max accuracy')
        if task_idx // 3 == 1:
            ax.set_xlabel('layer')
    plt.tight_layout()
    plt.show()


def plot_best_accuracy_per_layer(model_ids: list[str]):
    tasks = ["antonyms", "capitalize", "country-capital", "english-french", "present-past", "singular-plural"]
    fig, axss = plt.subplots(len(model_ids), 6, figsize=(12, 2*len(model_ids)), dpi=200, sharex='row', sharey=True)
    for model_idx, model_id in enumerate(model_ids):
        axs = axss if len(model_ids) == 1 else axss[model_idx]
        for task_idx, task in enumerate(tasks):
            # ax = axs[task_idx // 3][task_idx % 3]
            ax = axs[task_idx]
            # load and plot conceptor results
            df = read_all_csvs(f"../results/{model_id}_{task}_conceptor")
            df = df[df.experiment == "Average"]
            df_max = df.groupby('layer').max().reset_index()
            ax.plot(df_max.layer, df_max.final_accuracy, label='conceptor', lw=2)
            # load and plot addition results
            df = read_all_csvs(f"../results/{model_id}_{task}_addition")
            df = df[df.experiment == "Average"]
            df_max = df.groupby('layer').max().reset_index()
            ax.set_title(task)
            ax.plot(df_max.layer, df_max.final_accuracy, label='addition', lw=2)
            # plot baseline results
            ax.plot(df_max.layer, [0] * len(df_max.layer), label='baseline', lw=2, ls='--', color='red')
            if task_idx == 5:
                # put legend to the right outside of the plot
                ax.legend(bbox_to_anchor=(1.05, 1.05), loc='upper left')
            if task_idx == 0:
                model_dict = {"gpt-j-6B": "GPT-J (6B)", "gpt-neox-20b": "GPT-NeoX (20B)"}
                ax.set_ylabel(model_dict.get(model_id, model_id))
            ax.set_xlabel('layer')
    plt.tight_layout()
    plt.show()


def plot_hyperparameter_heatmap_addition(model_id: str):
    tasks = ["antonyms", "capitalize", "country-capital", "english-french", "present-past", "singular-plural"]
    fig, axs = plt.subplots(2, 3, figsize=(12, 16), dpi=200, sharex=True, sharey=True)
    model_dict = {"gpt-j-6B": "GPT-J (6B)", "gpt-neox-20b": "GPT-NeoX (20B)"}
    fig.suptitle(f"Additive steering of {model_dict[model_id]}")
    for task_idx, task in enumerate(tasks):
        df = read_all_csvs(f"../results/{model_id}_{task}_addition")
        df = df[df.experiment == "Average"]
        df_ = df.sort_values(by=['beta', 'layer'])['layer beta final_accuracy'.split(" ")]
        df_ = df_.groupby(['beta', 'layer']).mean().reset_index()
        df_ = df_.pivot(index='layer', columns='beta', values='final_accuracy')
        ax = axs[task_idx // 3][task_idx % 3]
        ax.set_title(task)
        sns.heatmap(df_, cmap='RdYlGn', annot=True, fmt=".1f", ax=ax)
        if task_idx % 3 == 0:
            ax.set_ylabel('layer')
        if task_idx // 3 == 1:
            ax.set_xlabel('beta')
    plt.tight_layout()
    plt.show()


def plot_hyperparameter_heatmap_conceptor(model_id: str, agg_over: str, agg_fn: str = "max"):
    assert agg_fn in ["max", "mean"], f"agg_fn must be one of ['max', 'mean'], got {agg_fn}"
    assert agg_over in ["beta", "aperture", "layer"], f"agg_over must be one of ['beta', 'aperture', 'layer'], got {agg_over}"
    tasks = ["antonyms", "capitalize", "country-capital", "english-french", "present-past", "singular-plural"]
    figsize = (12, 6) if agg_over == "layer" else (12, 16)
    fig, axs = plt.subplots(2, 3, figsize=figsize, dpi=200, sharex=True, sharey=True)
    model_dict = {"gpt-j-6B": "GPT-J (6B)", "gpt-neox-20b": "GPT-NeoX (20B)"}
    fig.suptitle(f"Conceptor steering of {model_dict.get(model_id, model_id)}")
    for task_idx, task in enumerate(tasks):
        x_name = [e for e in ["layer", "beta", "aperture"] if e != agg_over][0]
        y_name = [e for e in ["layer", "beta", "aperture"] if e != agg_over][1]
        df = read_all_csvs(f"../results/{model_id}_{task}_conceptor")
        df = df[df.experiment == "Average"]
        df_ = df.sort_values(by=[x_name, y_name])[["beta", "layer", "aperture", "final_accuracy"]]
        if agg_fn == "mean":
            df_ = df_.groupby([x_name, y_name]).mean().reset_index()
        elif agg_fn == "max":
            df_ = df_.groupby([x_name, y_name]).max().reset_index()
        df_ = df_.pivot(index=x_name, columns=y_name, values='final_accuracy')
        ax = axs[task_idx // 3][task_idx % 3]
        ax.set_title(task)
        sns.heatmap(df_, cmap='RdYlGn', annot=True, fmt=".1f", ax=ax)
        ax.set_ylabel(x_name)
        ax.set_xlabel(y_name)
    plt.tight_layout()
    plt.show()
