import numpy as np


def postprocess_results(raw_metrics: list[str]) -> np.ndarray:
    """Postprocesses the raw metrics by computing the mean of each metric across different runs.

    Parameters
    ----------
    raw_metrics : list[str]
        The raw metrics data.

    Returns
    -------
    np.ndarray
        The postprocessed mean metrics.
    """
    parsed_metrics = [[float(y) for y in x.split(",")] for x in raw_metrics]

    average_rows = [sum(metric) / len(metric) for metric in parsed_metrics]
    average = sum(average_rows) / len(average_rows)
    max_length = max(len(metric) for metric in parsed_metrics)

    processed_metrics = [([average] * (max_length - len(metric))) + metric for metric in parsed_metrics]
    mean_metrics = np.mean(np.array(processed_metrics), axis=0)

    return mean_metrics


def postprocess_results_bar_reversed(raw_metrics: list[str]) -> list[float]:
    """Postprocesses the raw metrics and calculates the reversed percentage change.

    Parameters
    ----------
    raw_metrics : list[str]
        The raw metric values.

    Returns
    -------
    list[float]
        The reversed percentage change values.
    """
    mean_metrics = postprocess_results(raw_metrics)

    baseline = mean_metrics[-1]
    mean_metrics = mean_metrics[:-1]
    reversed_percentage_change = list(reversed(np.divide(mean_metrics - baseline, baseline) * 100))

    return reversed_percentage_change


def postprocess_results_average_baselines(
        raw_baselines_chi2: list[str],
        raw_baselines_anova: list[str],
        raw_baselines_forward_selection: list[str],
        raw_baselines_backward_elimination: list[str]) -> tuple[list[float], list[float]]:
    """Postprocesses the raw baselines and calculates the reversed percentage change and baseline values.

    Parameters
    ----------
    raw_baselines_chi2 : list[str]
        The raw baselines values for Chi-Squared feature selection.
    raw_baselines_anova : list[str]
        The raw baselines values for ANOVA feature selection.
    raw_baselines_forward_selection : list[str]
        The raw baselines values for Forward Selection feature selection.
    raw_baselines_backward_elimination : list[str]
        The raw baselines values for Backward Elimination feature selection.

    Returns
    -------
    tuple[list[float], list[float]]
        A tuple containing the reversed percentage change values and the reversed baseline values.
    """
    comparison = postprocess_results(raw_metrics=raw_baselines_chi2 + raw_baselines_anova)
    baseline = postprocess_results(raw_metrics=raw_baselines_forward_selection + raw_baselines_backward_elimination)
    reversed_percentage_change = list(reversed(((comparison - baseline) / baseline) * 100))
    reversed_baseline = list(reversed(baseline))
    return reversed_percentage_change, reversed_baseline


def postprocess_runtime(raw_runtime: list[str]) -> float:
    """Postprocesses the raw runtime by computing the average runtime across different runs.

    Parameters
    ----------
    raw_runtime : list[str]
        The raw runtime data.

    Returns
    -------
    float
        The postprocessed average runtime.
    """
    parsed_runtime = [float(x) for x in raw_runtime if x != "nan"]
    average = sum(parsed_runtime) / len(parsed_runtime)
    return average
