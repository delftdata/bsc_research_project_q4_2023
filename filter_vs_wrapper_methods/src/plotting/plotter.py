from typing import Literal

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from processing.postprocessing import postprocess_results


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


def plot_metrics_matplotlib(
        raw_metrics_chi2: list[str],
        raw_metrics_anova: list[str],
        raw_metrics_forward_selection: list[str],
        raw_metrics_backward_elimination: list[str],
        model: str, x_label="Percentage of selected features", y_label="Accuracy"):

    metrics_chi2 = postprocess_results(raw_metrics_chi2)
    metrics_anova = postprocess_results(raw_metrics_anova)
    metrics_forward_selection = postprocess_results(raw_metrics_forward_selection)
    metrics_backward_elimination = postprocess_results(raw_metrics_backward_elimination)

    percentage_features = [i for i in range(10, 110, 10)]
    percentage_features = percentage_features[len(percentage_features) - len(metrics_chi2):]

    fig, ax = plt.subplots()

    ax.plot(percentage_features, metrics_chi2, label="chi2", marker="o")
    ax.plot(percentage_features, metrics_anova, label="anova", marker="*")
    ax.plot(percentage_features, metrics_forward_selection, label="forward_selection", marker="^")
    ax.plot(percentage_features, metrics_backward_elimination, label="backward elimination", marker="s")

    font_size_ticks = 7
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
