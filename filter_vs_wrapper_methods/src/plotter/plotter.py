import matplotlib.pyplot as plt

from processing.postprocessing import (postprocess_results,
                                       postprocess_results_average_baselines,
                                       postprocess_results_bar_reversed,
                                       postprocess_runtime)
from writer.writer import Writer

width = 11
height = 8.25
linewidth = 3
markersize = 12
font_size_ticks = 18
font_size_labels = 18
font_size_legend = 18
width_bar = 0.5
width_baseline_bar = 4.0


def plot_metrics_bar_leaderboard(
        raw_metrics_chi2: list[str],
        raw_metrics_anova: list[str],
        raw_metrics_forward_selection: list[str],
        raw_metrics_backward_elimination: list[str],
        leaderboard: str, model: str, y_label: str, baselines_path: str, x_label="Percentage of selected features",
        save_percentages=True):
    """Plots the metrics for feature selection techniques in a bar chart based on the leaderboard.

    Parameters
    ----------
    raw_metrics_chi2 : list[str]
        The raw metric values for Chi-Squared feature selection.
    raw_metrics_anova : list[str]
        The raw metric values for ANOVA feature selection.
    raw_metrics_forward_selection : list[str]
        The raw metric values for Forward Selection feature selection.
    raw_metrics_backward_elimination : list[str]
        The raw metric values for Backward Elimination feature selection.
    leaderboard : str
        The leaderboard to determine the selection metrics ("best", "second", "third", "worst").
    model : str
        The name of the model.
    y_label : str
        The label for the y-axis.
    baselines_path : str
        The path to save the baseline metrics.
    x_label : str, optional
        The label for the x-axis (default: "Percentage of selected features").
    save_percentages : bool, optional
        Whether to save the percentage values (default: True).

    Returns
    -------
        The generated Matplotlib figure.
    """
    y_label = f"{y_label} improvement (%)"

    reversed_percentage_change_chi2 = postprocess_results_bar_reversed(raw_metrics_chi2)
    reversed_percentage_change_anova = postprocess_results_bar_reversed(raw_metrics_anova)
    reversed_percentage_change_forward_selection = postprocess_results_bar_reversed(raw_metrics_forward_selection)
    reversed_percentage_change_backward_elimination = postprocess_results_bar_reversed(raw_metrics_backward_elimination)

    selection_metrics_dictionary = {
        "best": list(reversed(list(max(a, b, c, d) for a, b, c, d in zip(
            reversed_percentage_change_chi2, reversed_percentage_change_anova, reversed_percentage_change_forward_selection, reversed_percentage_change_backward_elimination)))),
        "second": list(reversed(list(sorted([a, b, c, d], reverse=True)[1] for a, b, c, d in zip(
            reversed_percentage_change_chi2, reversed_percentage_change_anova, reversed_percentage_change_forward_selection, reversed_percentage_change_backward_elimination)))),
        "third": list(reversed(list(sorted([a, b, c, d], reverse=True)[2] for a, b, c, d in zip(
            reversed_percentage_change_chi2, reversed_percentage_change_anova, reversed_percentage_change_forward_selection, reversed_percentage_change_backward_elimination)))),
        "worst": list(reversed(list(min(a, b, c, d) for a, b, c, d in zip(
            reversed_percentage_change_chi2, reversed_percentage_change_anova, reversed_percentage_change_forward_selection, reversed_percentage_change_backward_elimination)))),
    }

    selection_metrics = selection_metrics_dictionary[leaderboard]

    fig, ax = plt.subplots(figsize=(width, height))

    percentage_features = list(range(10, 100, 10))
    min_percentage_index = len(percentage_features) - len(selection_metrics)
    percentage_features = percentage_features[min_percentage_index:]

    percentage_change_chi2 = reversed(reversed_percentage_change_chi2)
    percentage_change_anova = reversed(reversed_percentage_change_anova)
    percentage_change_forward_selection = reversed(reversed_percentage_change_forward_selection)
    percentage_change_backward_elimination = reversed(reversed_percentage_change_backward_elimination)

    if save_percentages:
        Writer.write_to_file(baselines_path, "chi2.txt", ",".join([str(x) for x in percentage_change_chi2]))
        Writer.write_to_file(baselines_path, "anova.txt", ",".join([str(x) for x in percentage_change_anova]))
        Writer.write_to_file(baselines_path, "forward_selection.txt", ",".join(
            [str(x) for x in percentage_change_forward_selection]))
        Writer.write_to_file(baselines_path, "backward_elimination.txt", ",".join(
            [str(x) for x in percentage_change_backward_elimination]))

    percentage_change_chi2 = [metric if metric == selection_metrics[i]
                              else 0.0 for i, metric in enumerate(percentage_change_chi2)]
    percentage_change_anova = [metric if metric == selection_metrics[i]
                               else 0.0 for i, metric in enumerate(percentage_change_anova)]
    percentage_change_forward_selection = [metric if metric == selection_metrics[i] else 0.0 for i,
                                           metric in enumerate(percentage_change_forward_selection)]
    percentage_change_backward_elimination = [metric if metric == selection_metrics[i] else 0.0 for i,
                                              metric in enumerate(percentage_change_backward_elimination)]

    ax.bar(percentage_features, percentage_change_chi2, align="center", label="chi2", width=width_baseline_bar)
    ax.bar(percentage_features, percentage_change_anova, align="center", label="anova", width=width_baseline_bar)
    ax.bar(percentage_features, percentage_change_forward_selection, align="center",
           label="forward_selection", width=width_baseline_bar)
    ax.bar(percentage_features, percentage_change_backward_elimination, align="center",
           label="backward_elimination", width=width_baseline_bar)
    ax.axhline(y=0.0, color="purple", linestyle="-", label="Baseline: all features")

    plt.xticks(percentage_features, fontsize=font_size_ticks, weight="bold")
    plt.yticks(fontsize=font_size_ticks, weight="bold")

    plt.xlabel(x_label, fontsize=font_size_labels, weight="bold")
    plt.ylabel(y_label, fontsize=font_size_labels, weight="bold")
    plt.title(model, fontsize=font_size_labels, weight="bold")

    plt.legend(fontsize=font_size_legend)

    return fig


def plot_baselines_bar(raw_baselines_chi2: list[str],
                       raw_baselines_anova: list[str],
                       raw_baselines_forward_selection: list[str],
                       raw_baselines_backward_elimination: list[str],
                       average_baselines_path: str,
                       title="Average percentage change. Baseline: wrapper methods.",
                       x_label="Percentage of selected features", y_label="Percentage change (%)",
                       save_percentages=True):
    """Plots the average percentage change of baseline methods in a bar chart.

    Parameters
    ----------
    raw_baselines_chi2 : list[str]
        The raw baseline values for Chi-Squared feature selection.
    raw_baselines_anova : list[str]
        The raw baseline values for ANOVA feature selection.
    raw_baselines_forward_selection : list[str]
        The raw baseline values for Forward Selection feature selection.
    raw_baselines_backward_elimination : list[str]
        The raw baseline values for Backward Elimination feature selection.
    average_baselines_path : str
        The path to save the average baseline metrics.
    title : str, optional
        The title of the plot (default: "Average percentage change. Baseline: wrapper methods.").
    x_label : str, optional
        The label for the x-axis (default: "Percentage of selected features").
    y_label : str, optional
        The label for the y-axis (default: "Percentage change (%)").
    save_percentages : bool, optional
        Whether to save the percentage values (default: True).

    Returns
    -------
        The generated Matplotlib figure.
    """
    reversed_percentage_change_filter, reversed_baseline_wrapper = postprocess_results_average_baselines(
        raw_baselines_chi2, raw_baselines_anova, raw_baselines_forward_selection, raw_baselines_backward_elimination)

    fig, ax = plt.subplots(figsize=(width, height))

    percentage_change_filter = list(reversed(reversed_percentage_change_filter))
    baseline_wrapper = list(reversed(reversed_baseline_wrapper))

    if save_percentages:
        Writer.write_to_file(average_baselines_path, "percentage_change_filter.txt",
                             ",".join([str(x) for x in percentage_change_filter]))
        Writer.write_to_file(average_baselines_path, "baseline_wrapper.txt",
                             ",".join([str(x) for x in baseline_wrapper]))

    percentage_features = list(range(10, 110, 10))

    ax.bar(percentage_features, percentage_change_filter, align="center",
           label="filter_methods", width=width_baseline_bar)
    ax.axhline(y=0.0, color="purple", linestyle="-", label="Baseline: wrapper_methods")

    plt.xticks(percentage_features, fontsize=font_size_ticks, weight="bold")
    plt.yticks(fontsize=font_size_ticks, weight="bold")

    plt.xlabel(x_label, fontsize=font_size_labels, weight="bold")
    plt.ylabel(y_label, fontsize=font_size_labels, weight="bold")
    plt.title(title, fontsize=font_size_labels, weight="bold")

    plt.legend(fontsize=font_size_legend)

    return fig


def plot_runtime(
        raw_runtime_chi2: list[str],
        raw_runtime_anova: list[str],
        raw_runtime_forward_selection: list[str],
        raw_runtime_backward_elimination: list[str],
        x_label="Feature selection techniques", y_label="Runtime in minutes",
        title="Average runtime of feature selection techniques"):
    """Plots the average runtime of feature selection techniques.

    Parameters
    ----------
    raw_runtime_chi2 : list[str]
        The raw runtime values for Chi-Squared feature selection.
    raw_runtime_anova : list[str]
        The raw runtime values for ANOVA feature selection.
    raw_runtime_forward_selection : list[str]
        The raw runtime values for Forward Selection feature selection.
    raw_runtime_backward_elimination : list[str]
        The raw runtime values for Backward Elimination feature selection.
    x_label : str, optional
        The label for the x-axis (default: "Feature selection techniques").
    y_label : str, optional
        The label for the y-axis (default: "Runtime in minutes").
    title : str, optional
        The title of the plot (default: "Average runtime of feature selection techniques").

    Returns
    -------
        The generated Matplotlib figure.
    """
    runtime_chi2 = postprocess_runtime(raw_runtime_chi2)
    runtime_anova = postprocess_runtime(raw_runtime_anova)
    runtime_forward_selection = postprocess_runtime(raw_runtime_forward_selection)
    runtime_backward_elimination = postprocess_runtime(raw_runtime_backward_elimination)
    methods = ["Chi-Squared", "ANOVA", "Forward Selection", "Backward Elimination"]
    runtime_methods = [runtime_chi2, runtime_anova, runtime_forward_selection, runtime_backward_elimination]
    runtime_minutes_methods = [x / 60.0 for x in runtime_methods]

    fig, ax = plt.subplots(figsize=(width, height))

    ax.bar(methods, runtime_minutes_methods, align="center", width=width_bar)

    plt.xticks(fontsize=font_size_ticks * 0.7, weight="bold")
    plt.yticks(fontsize=font_size_ticks, weight="bold")

    plt.xlabel(x_label, fontsize=font_size_labels, weight="bold")
    plt.ylabel(y_label, fontsize=font_size_labels, weight="bold")
    plt.title(title, fontsize=font_size_labels, weight="bold")

    return fig


def plot_metrics(
        raw_metrics_chi2: list[str],
        raw_metrics_anova: list[str],
        raw_metrics_forward_selection: list[str],
        raw_metrics_backward_elimination: list[str],
        model: str, x_label="Percentage of selected features", y_label="Accuracy"):
    """Plots the metrics for different feature selection techniques.

    Parameters
    ----------
    raw_metrics_chi2 : list[str]
        The raw metric values for Chi-Squared feature selection.
    raw_metrics_anova : list[str]
        The raw metric values for ANOVA feature selection.
    raw_metrics_forward_selection : list[str]
        The raw metric values for Forward Selection feature selection.
    raw_metrics_backward_elimination : list[str]
        The raw metric values for Backward Elimination feature selection.
    model : str
        The model for which the metrics are collected.
    x_label : str, optional
        The label for the x-axis, default: "Percentage of selected features".
    y_label : str, optional
        The label for the y-axis, default: "Accuracy".

    Returns
    -------
        The generated Matplotlib figure.
    """
    metrics_chi2 = postprocess_results(raw_metrics_chi2) if raw_metrics_chi2 else []
    metrics_anova = postprocess_results(raw_metrics_anova) if raw_metrics_anova else []
    metrics_forward_selection = postprocess_results(
        raw_metrics_forward_selection) if raw_metrics_forward_selection else []
    metrics_backward_elimination = postprocess_results(
        raw_metrics_backward_elimination) if raw_metrics_backward_elimination else []

    percentage_features = list(range(10, 110, 10))

    fig, ax = plt.subplots(figsize=(width, height))

    if raw_metrics_chi2:
        min_percentage_index = len(percentage_features) - len(metrics_chi2)
        ax.plot(percentage_features[min_percentage_index:], metrics_chi2,
                label="chi2", marker="s", linewidth=linewidth, markersize=markersize)
    if raw_metrics_anova:
        min_percentage_index = len(percentage_features) - len(metrics_anova)
        ax.plot(percentage_features[min_percentage_index:], metrics_anova, label="anova",
                marker="o", linewidth=linewidth, markersize=markersize)
    if raw_metrics_forward_selection:
        min_percentage_index = len(percentage_features) - len(metrics_forward_selection)
        ax.plot(percentage_features[min_percentage_index:], metrics_forward_selection,
                label="forward_selection", marker="^", linewidth=linewidth, markersize=markersize)
    if raw_metrics_backward_elimination:
        min_percentage_index = len(percentage_features) - len(metrics_backward_elimination)
        ax.plot(percentage_features[min_percentage_index:], metrics_backward_elimination,
                label="backward_elimination",  marker="X", linewidth=linewidth, markersize=markersize)

    plt.xticks(percentage_features, fontsize=font_size_ticks, weight="bold")
    plt.yticks(fontsize=font_size_ticks, weight="bold")

    plt.xlabel(x_label, fontsize=font_size_labels, weight="bold")
    plt.ylabel(y_label, fontsize=font_size_labels, weight="bold")
    plt.title(model, fontsize=font_size_labels, weight="bold")

    plt.legend(fontsize=font_size_legend)

    return fig
