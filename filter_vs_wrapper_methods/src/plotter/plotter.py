import matplotlib.pyplot as plt

from processing.postprocessing import postprocess_results, postprocess_runtime

width = 11
height = 8.25
linewidth = 3
markersize = 12
font_size_ticks = 18
font_size_labels = 18
font_size_legend = 18
width_bar = 0.5


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


def plot_metrics_3D(raw_metrics_chi2: list[str],
                    raw_metrics_anova: list[str],
                    raw_metrics_forward_selection: list[str],
                    raw_metrics_backward_elimination: list[str],
                    raw_runtime_chi2: list[str],
                    raw_runtime_anova: list[str],
                    raw_runtime_forward_selection: list[str],
                    raw_runtime_backward_elimination: list[str],
                    model: str, x_label="Percentage of selected features", y_label="Accuracy",
                    z_label="Runtime feature selection"):
    """Plots a 3D graph showing the metrics and quantifying also the runtime for different feature selection methods.

    Parameters:
        raw_metrics_chi2 (list[str]):
            Raw metrics data for the Chi-Squared feature selection method.
        raw_metrics_anova (list[str]):
            Raw metrics data for the ANOVA feature selection method.
        raw_metrics_forward_selection (list[str]):
            Raw metrics data for the Forward Selection feature selection method.
        raw_metrics_backward_elimination (list[str]):
            Raw metrics data for the Backward Elimination feature selection method.
        raw_runtime_chi2 (list[str]):
            Raw runtime data for the Chi-Squared feature selection method.
        raw_runtime_anova (list[str]):
            Raw runtime data for the ANOVA feature selection method.
        raw_runtime_forward_selection (list[str]):
            Raw runtime data for the Forward Selection feature selection method.
        raw_runtime_backward_elimination (list[str]):
            Raw runtime data for the Backward Elimination feature selection method.
        model (str):
            The name of the model.
        x_label (str, optional):
            Label for the x-axis. Defaults to "Percentage of selected features".
        y_label (str, optional):
            Label for the y-axis. Defaults to "Accuracy".
        z_label (str, optional):
            Label for the z-axis. Defaults to "Runtime feature selection".

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

    runtime_chi2 = [postprocess_runtime(raw_runtime_chi2)] * len(metrics_chi2)
    runtime_anova = [postprocess_runtime(raw_runtime_anova)] * len(metrics_anova)
    runtime_forward_selection = [postprocess_runtime(
        raw_runtime_forward_selection)] * len(metrics_forward_selection)
    runtime_backward_elimination = [postprocess_runtime(
        raw_runtime_backward_elimination)] * len(metrics_backward_elimination)

    percentage_features = list(range(10, 110, 10))

    fig, ax = plt.subplots(figsize=(width, height))
    ax = plt.axes(projection="3d")

    if raw_metrics_chi2:
        min_percentage_index = len(percentage_features) - len(metrics_chi2)
        ax.plot(percentage_features[min_percentage_index:], metrics_chi2, runtime_chi2,
                label="chi2", marker="s", linewidth=linewidth, markersize=markersize)

    if raw_metrics_anova:
        min_percentage_index = len(percentage_features) - len(metrics_anova)
        ax.plot(percentage_features[min_percentage_index:], metrics_anova, runtime_anova,  label="anova",
                marker="o", linewidth=linewidth, markersize=markersize)

    if raw_metrics_forward_selection:
        min_percentage_index = len(percentage_features) - len(metrics_forward_selection)
        ax.plot(percentage_features[min_percentage_index:], metrics_forward_selection, runtime_forward_selection,
                label="forward_selection", marker="^", linewidth=linewidth, markersize=markersize)

    if raw_metrics_backward_elimination:
        min_percentage_index = len(percentage_features) - len(metrics_backward_elimination)
        ax.plot(percentage_features[min_percentage_index:], metrics_backward_elimination, runtime_backward_elimination,
                label="backward_elimination", marker="X", linewidth=linewidth, markersize=markersize)

    plt.xticks(percentage_features, weight="bold")
    plt.yticks(weight="bold")

    plt.xlabel(x_label, fontsize=font_size_labels - 2, weight="bold")
    plt.ylabel(y_label, fontsize=font_size_labels - 4, weight="bold")
    ax.set_zlabel(z_label, fontsize=font_size_labels - 2, weight="bold")  # type: ignore
    plt.title(model, fontsize=font_size_labels, weight="bold")

    plt.legend(fontsize=font_size_legend - 2, loc="upper left", framealpha=0.2)

    return fig
