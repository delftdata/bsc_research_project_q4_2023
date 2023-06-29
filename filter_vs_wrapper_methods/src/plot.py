import json
import os

import matplotlib.pyplot as plt

from plotter.plotter import (plot_metrics, plot_metrics_3D, plot_runtime,
                             plot_runtime_features_scatter_plot)

models = ["GBM", "LR", "RF", "XGB", "SVM"]


def main():
    """Plots the experimental results of the research project `Automatic Feature Discovery: A comparative study between
    filter and wrapper feature selection techniques`.

    Raises
    ------
    OSError
        If there is a mismatch between the provided `arguments_plot.json` path and its actual path.
    KeyError
        If the provided plot_type is invalid.
    """
    classification_datasets = ["arrhythmia", "bank_marketing", "breast_cancer", "census_income",
                               "internet_advertisements", "steel_plates_faults", "character_font_images"]
    regression_datasets = ["bike_sharing", "housing_prices", "nasa_numeric"]
    y_label_regression = "Root Mean Squared Error"

    with open("arguments_plot.json", "r", encoding="utf-8") as json_file:
        arguments = json.load(json_file)

        experiment_name: str = arguments["experiment_name"]
        print(f"Experiment name: {experiment_name}")

        dataset: str = arguments["dataset"]
        print(f"Dataset: {dataset}")
        y_label = "Accuracy" if dataset in classification_datasets else y_label_regression

        plot_type: str = arguments["plot_type"]
        print(f"Plot type: {plot_type}")

        plot_3D: bool = arguments["3D"]
        print(f"3D: {plot_3D}")

        if plot_type == "results":
            plot_experiments(experiment_name, dataset, y_label, plot_3D)
        elif plot_type == "all_results":
            for dataset in (classification_datasets + regression_datasets):
                y_label = "Accuracy" if dataset in classification_datasets else y_label_regression
                plot_experiments(experiment_name, dataset, y_label, plot_3D)
        elif plot_type == "average_runtime":
            plot_average_runtime(datasets=classification_datasets + regression_datasets)
        elif plot_type == "runtime_features":
            plot_runtime_features(experiment_name, classification_datasets + regression_datasets)
        else:
            raise KeyError(f"The plot_type: {plot_type} is not valid.")


def plot_runtime_features(experiment_name: str, datasets: list[str]):
    """Plots and saves a scatter plot of average runtime for different feature selection techniques.

    Parameters
    ----------
    experiment_name : str
        The name of the experiment.
    datasets : list[str]
        The list of datasets to process.
    """
    average_runtime_chi2, average_runtime_anova, average_runtime_forward_selection, average_runtime_backward_elimination = [], [], [], []
    amount_features = []
    for dataset in datasets:
        for preprocessing_variation in ("", "no_normalization", "median"):
            if not preprocessing_variation:
                results_path = f"results/{experiment_name}/{dataset}"
            else:
                results_path = f"results/{experiment_name}/{dataset}/{preprocessing_variation}"

            for dataset in datasets:
                runtime_path = f"{results_path}/runtime"
                features_path = f"{results_path}/selected_features/forward_selection.txt"
                data_types = ["categorical", "discrete", "continuous"]

                for data_type in data_types:
                    if experiment_name == "experiment4":
                        runtime_path = f"{results_path}/{data_type}/runtime"
                        features_path = f"{results_path}/{data_type}/selected_features/forward_selection.txt"

                    try:
                        runtime_chi2 = open_raw_average_runtime(runtime_path, "chi2")
                        runtime_anova = open_raw_average_runtime(runtime_path, "anova")
                        runtime_forward_selection = open_raw_average_runtime(runtime_path, "forward_selection")
                        runtime_backward_elimination = open_raw_average_runtime(runtime_path, "backward_elimination")
                        features = open_raw_features(features_path)

                        average_runtime_chi2.append(runtime_chi2)
                        average_runtime_anova.append(runtime_anova)
                        average_runtime_forward_selection.append(runtime_forward_selection)
                        average_runtime_backward_elimination.append(runtime_backward_elimination)
                        amount_features.append(features)
                    except OSError as error:
                        print(f"Runtime: {error}")

                    if experiment_name != "experiment4":
                        break

    algorithm_plot = plot_runtime_features_scatter_plot(
        average_runtime_chi2, average_runtime_anova, average_runtime_forward_selection,
        average_runtime_backward_elimination, amount_features)

    algorithm_plot.savefig(f"results/{experiment_name}/runtime_features.png")
    plt.close(algorithm_plot)


def plot_average_runtime(datasets: list[str]):
    """Plots the average runtime across all experiments of different feature selection methods for the given datasets.

    Parameters
    ----------
    datasets : list[str]
        The list of datasets to plot the average runtime for.
    """
    experiments = ["experiment1", "experiment2", "experiment3", "experiment4"]
    raw_runtime_chi2, raw_runtime_anova, raw_runtime_forward_selection, raw_runtime_backward_elimination = [], [], [], []

    for experiment in experiments:
        results_path = f"results/{experiment}"
        for dataset in datasets:
            runtime_path = f"{results_path}/{dataset}/runtime"
            data_types = ["categorical", "discrete", "continuous"]

            for data_type in data_types:
                if experiment == "experiment4":
                    runtime_path = f"{results_path}/{dataset}/{data_type}/runtime"

                try:
                    raw_runtime_chi2 += open_raw_runtime(runtime_path, "chi2")
                    raw_runtime_anova += open_raw_runtime(runtime_path, "anova")
                    raw_runtime_forward_selection += open_raw_runtime(runtime_path, "forward_selection")
                    raw_runtime_backward_elimination += open_raw_runtime(runtime_path, "backward_elimination")
                except OSError as error:
                    print(f"Runtime: {error}")

                if experiment != "experiment4":
                    break

    plot_path = "results/runtime"
    if not os.path.isdir(plot_path):
        os.makedirs(plot_path)

    algorithm_plot = plot_runtime(raw_runtime_chi2, raw_runtime_anova,
                                  raw_runtime_forward_selection, raw_runtime_backward_elimination)

    algorithm_plot.savefig(f"{plot_path}/average_runtime.png")
    plt.close(algorithm_plot)


def plot_experiments(experiment_name: str, dataset: str, y_label="Accuracy", plot_3D=False):
    """Plots the results for a specific experiment and dataset.

    Parameters
    ----------
    experiment_name : str
        The name of the experiment.
    dataset : str
        The name of the dataset.
    y_label : str, optional
        The label for the y-axis of the plot (default: "Accuracy").
    plot_3D : bool, optional
        Whether to plot 3D graphs (default: False).
    """
    for preprocessing_variation in ("", "no_normalization", "median"):
        if not preprocessing_variation:
            results_path = f"results/{experiment_name}/{dataset}"
        else:
            results_path = f"results/{experiment_name}/{dataset}/{preprocessing_variation}"
        if experiment_name == "experiment4":
            data_types = ["categorical", "discrete", "continuous"]
            for data_type in data_types:
                if plot_3D:
                    plot_results_3D(f"{results_path}/{data_type}", y_label)
                else:
                    plot_results(f"{results_path}/{data_type}", y_label)
        else:
            if plot_3D:
                plot_results_3D(results_path, y_label)
            else:
                plot_results(results_path, y_label)


def plot_results(results_path: str, y_label="Accuracy"):
    """Plots the results for a specific results path.

    Parameters
    ----------
    results_path : str
        The path to the directory containing the results.
    y_label : str, optional
        The label for the y-axis of the plot (default: "Accuracy").
    """
    for model in models:
        if not os.path.isfile(f"{results_path}/chi2/{model}.txt") \
                and not os.path.isfile(f"{results_path}/anova/{model}.txt") \
                and not os.path.isfile(f"{results_path}/forward_selection/{model}.txt") \
                and not os.path.isfile(f"{results_path}/backward_elimination/{model}.txt"):
            continue
        try:
            raw_metrics_chi2 = try_opening_raw_metrics(results_path, "chi2", model)
            raw_metrics_anova = try_opening_raw_metrics(results_path, "anova", model)
            raw_metrics_forward_selection = try_opening_raw_metrics(results_path, "forward_selection", model)
            raw_metrics_backward_elimination = try_opening_raw_metrics(results_path, "backward_elimination", model)

            algorithm_plot = plot_metrics(
                raw_metrics_chi2, raw_metrics_anova, raw_metrics_forward_selection, raw_metrics_backward_elimination,
                model, y_label=y_label)

            algorithm_plot.savefig(f"{results_path}/{model}.png")
            plt.close(algorithm_plot)
        except Exception as error:
            print(f"Failed plotting or saving: {error}")


def plot_results_3D(results_path: str, y_label="Accuracy"):
    """Plots 3D graphs results for a specific results path.

    Parameters:
        results_path (str):
            The path to the results.
        y_label (str, optional):
            Label for the y-axis. Defaults to "Accuracy".
    """
    results_path = f"{results_path}"
    runtime_path = f"{results_path}/runtime"

    for model in models:
        if not os.path.isfile(f"{results_path}/chi2/{model}.txt") \
                and not os.path.isfile(f"{results_path}/anova/{model}.txt") \
                and not os.path.isfile(f"{results_path}/forward_selection/{model}.txt") \
                and not os.path.isfile(f"{results_path}/backward_elimination/{model}.txt"):
            continue
        try:
            raw_metrics_chi2 = try_opening_raw_metrics(results_path, "chi2", model)
            raw_metrics_anova = try_opening_raw_metrics(results_path, "anova", model)
            raw_metrics_forward_selection = try_opening_raw_metrics(results_path, "forward_selection", model)
            raw_metrics_backward_elimination = try_opening_raw_metrics(results_path, "backward_elimination", model)

            raw_runtime_chi2 = open_raw_runtime(runtime_path, "chi2")
            raw_runtime_anova = open_raw_runtime(runtime_path, "anova")
            raw_runtime_forward_selection = open_raw_runtime(runtime_path, "forward_selection")
            raw_runtime_backward_elimination = open_raw_runtime(runtime_path, "backward_elimination")

            algorithm_plot = plot_metrics_3D(
                raw_metrics_chi2, raw_metrics_anova, raw_metrics_forward_selection, raw_metrics_backward_elimination,
                raw_runtime_chi2, raw_runtime_anova, raw_runtime_forward_selection, raw_runtime_backward_elimination,
                model, y_label=y_label)

            path_3D = f"{results_path}/3D"
            if not os.path.isdir(path_3D):
                os.mkdir(path_3D)

            algorithm_plot.savefig(f"{path_3D}/{model}.png")
            plt.close(algorithm_plot)
        except Exception as error:
            print(f"Failed plotting or saving: {error}")


def open_raw_metrics(results_path: str, method: str, model: str) -> list[str]:
    """Opens and reads the raw metrics from a file.

    Parameters
    ----------
    results_path : str
        The path to the directory containing the results.
    method : str
        The method used for feature selection.
    model : str
        The name of the model.

    Returns
    -------
    list[str]
        A list of raw metrics read from the file.

    Raises
    ------
    OSError
        If there is an error while opening or reading the file.
    """
    with open(f"{results_path}/{method}/{model}.txt", "r", encoding="utf-8") as lines:
        return [line.strip() for line in lines]


def open_raw_runtime(runtime_path: str, method: str) -> list[str]:
    """Opens and reads the raw runtime values from a file.

    Parameters
    ----------
    runtime_path : str
        The path to the directory containing the runtime results.
    method : str
        The method used for feature selection.

    Returns
    -------
    list[str]
        A list of raw runtime values read from the file.

    Raises
    ------
    OSError
        If there is an error while opening or reading the file.
    """
    with open(f"{runtime_path}/{method}.txt", "r", encoding="utf-8") as lines:
        return [line.strip() for line in lines]


def open_raw_average_runtime(runtime_path: str, method: str) -> float:
    """Opens the raw average runtime data for a specific method from a given file path.

    Parameters
    ----------
    runtime_path : str
        The file path to the runtime data.
    method : str
        The specific method for which to retrieve the average runtime.

    Returns
    -------
    float
        The average runtime value for the specified method.
    """
    raw_runtime = open_raw_runtime(runtime_path, method)
    runtime = [float(x) for x in raw_runtime if x != "nan"]
    return sum(runtime) / len(runtime) if runtime else 0


def open_raw_features(features_path: str) -> int:
    """Opens the file at the given path and counts the number of non-empty lines.

    Parameters
    ----------
    features_path : str
        The file path to the features data.

    Returns
    -------
    int
        The number of non-empty lines in the file.
    """
    with open(features_path, "r", encoding="utf-8") as lines:
        return len([line for line in lines if line])


def try_opening_raw_metrics(results_path: str, method: str, model: str) -> list[str]:
    """Attempts to open and read the raw metrics values from a file, handling any errors.

    Parameters
    ----------
    results_path : str
        The path to the directory containing the results.
    method : str
        The method used for feature selection.
    model : str
        The model for which the metrics are collected.

    Returns
    -------
    list[str]
        A list of raw metrics values read from the file, or an empty list if there was an error.

    Raises
    ------
    OSError
        If there is an error while opening or reading the file.
    """
    raw_metrics = []
    try:
        raw_metrics = open_raw_metrics(results_path, method, model)
    except OSError as error:
        print(error)
    return raw_metrics


if __name__ == "__main__":
    main()
