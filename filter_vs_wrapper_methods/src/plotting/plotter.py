from typing import Literal

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from processing.postprocessing import postprocess_results, postprocess_runtime


def plot_metrics_sea_born(performance: dict[str, list[float]],
                          scoring: Literal["Accuracy", "Mean Squared Error"],
                          x_axis="Percentage of selected features",
                          legend="Method"):

    data_frame_dictionary: dict[str, list] = dict()

    for (key_percentage_or_method, values_percentage_or_metric) in performance.items():
        data_frame_dictionary[key_percentage_or_method] = values_percentage_or_metric

    df = pd.DataFrame(data_frame_dictionary)
    df_melt = pd.melt(df, [x_axis])
    df_melt.rename(columns={"value": scoring, "variable": legend}, inplace=True)

    return sns.lineplot(x=x_axis, y=scoring, hue=legend, data=df_melt)


def plot_runtime_matplotlib(
        raw_runtime_chi2: list[str],
        raw_runtime_anova: list[str],
        raw_runtime_forward_selection: list[str],
        raw_runtime_backward_elimination: list[str],
        x_label="Feature selection techniques", y_label="Runtime in minutes",
        title="Average runtime of feature selection techniques"):
    runtime_chi2 = postprocess_runtime(raw_runtime_chi2)
    runtime_anova = postprocess_runtime(raw_runtime_anova)
    runtime_forward_selection = postprocess_runtime(raw_runtime_forward_selection)
    runtime_backward_elimination = postprocess_runtime(raw_runtime_backward_elimination)
    methods = ["Chi-Squared", "ANOVA", "Forward Selection", "Backward Elimination"]
    runtime_methods = [runtime_chi2, runtime_anova, runtime_forward_selection, runtime_backward_elimination]
    runtime_minutes_methods = [x / 60.0 for x in runtime_methods]

    fig, ax = plt.subplots()

    ax.bar(methods, runtime_minutes_methods, align="center", width=0.5)

    font_size_ticks = 7
    plt.xticks(fontsize=font_size_ticks)
    plt.yticks(fontsize=font_size_ticks)

    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.title(title)
    return fig


def plot_metrics_matplotlib(
        raw_metrics_chi2: list[str],
        raw_metrics_anova: list[str],
        raw_metrics_forward_selection: list[str],
        raw_metrics_backward_elimination: list[str],
        model: str, x_label="Percentage of selected features", y_label="Accuracy"):

    metrics_chi2 = postprocess_results(raw_metrics_chi2) if raw_metrics_chi2 else []
    metrics_anova = postprocess_results(raw_metrics_anova) if raw_metrics_anova else []
    metrics_forward_selection = postprocess_results(
        raw_metrics_forward_selection) if raw_metrics_forward_selection else []
    metrics_backward_elimination = postprocess_results(
        raw_metrics_backward_elimination) if raw_metrics_backward_elimination else []

    percentage_features = [i for i in range(10, 110, 10)]
    percentage_features = percentage_features[len(percentage_features) - len(metrics_chi2):]

    fig, ax = plt.subplots()

    if raw_metrics_chi2:
        ax.plot(percentage_features, metrics_chi2, label="chi2", marker="o")
    if raw_metrics_anova:
        ax.plot(percentage_features, metrics_anova, label="anova", marker="*")
    if raw_metrics_forward_selection:
        ax.plot(percentage_features, metrics_forward_selection, label="forward_selection", marker="^")
    if raw_metrics_backward_elimination:
        ax.plot(percentage_features, metrics_backward_elimination, label="backward elimination", marker="s")

    font_size_ticks = 10
    plt.xticks(percentage_features, fontsize=font_size_ticks)
    plt.yticks(fontsize=font_size_ticks)
    # plt.yticks(
    #     np.unique(
    #         np.concatenate(
    #             (metrics_chi2, metrics_anova, metrics_forward_selection, metrics_backward_elimination),
    #             axis=None)))

    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.title(model)
    plt.legend()

    return fig
