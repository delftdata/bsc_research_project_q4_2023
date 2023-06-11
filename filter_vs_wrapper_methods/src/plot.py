import os

import matplotlib.pyplot as plt

from plotting.plotter import plot_metrics_matplotlib, plot_runtime_matplotlib


def main():
    # plot_results("experiment2", "breast_cancer")
    # plot_results("experiment2", "steel_plates_faults")
    # plot_results("experiment2", "bank_marketing")
    # plot_experiments("experiment4", "arrhythmia")
    # plot_experiments("experiment4", "bank_marketing")
    # plot_experiments("experiment4", "BikeSharing", y_label="Root Mean Squared Error")
    # plot_experiments("experiment4", "breast_cancer")
    # plot_experiments("experiment4", "census_income")
    # plot_experiments("experiment4", "connect-4")
    # plot_experiments("experiment4", "housing_prices", y_label="Root Mean Squared Error")
    # plot_experiments("experiment4", "InternetAdvertisements")
    # plot_experiments("experiment4", "nasa_numeric", y_label="Root Mean Squared Error")
    # plot_experiments("experiment1", "steel_plates_faults")
    classification_datasets = ["arrhythmia", "bank_marketing", "breast_cancer", "census_income", "steel_plates_faults"]
    regression_datasets = ["housing_prices", "BikeSharing", "nasa_numeric"]
    # y_label_regression = "Average Root Mean Squared Error"
    # # plot_average_results_experiment2(classification_datasets)
    # # plot_average_results_experiment2(regression_datasets, y_label=y_label_regression)
    # plot_average_results_experiment4(classification_datasets)
    # plot_average_results_experiment4(regression_datasets, y_label=y_label_regression)
    plot_average_runtime(datasets=(classification_datasets + regression_datasets))


def plot_average_runtime(datasets: list[str]):
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
                except OSError as e:
                    print(f"Runtime: {e}")

                if experiment != "experiment4":
                    break

    plot_path = "results/runtime"
    if not os.path.isdir(plot_path):
        os.makedirs(plot_path)

    algorithm_plot = plot_runtime_matplotlib(raw_runtime_chi2, raw_runtime_anova,
                                             raw_runtime_forward_selection, raw_runtime_backward_elimination)

    algorithm_plot.savefig(f"{plot_path}/average_runtime.png")
    plt.close(algorithm_plot)


def plot_average_results_experiment2(datasets: list[str], y_label="Average Accuracy"):
    results_path_experiment = "results/experiment2"
    models = ["GBM", "LR", "RF", "XGB"]
    for model in models:
        raw_metrics_chi2, raw_metrics_anova, raw_metrics_forward_selection, raw_metrics_backward_elimination = [], [], [], []
        for dataset in datasets:
            results_path = f"{results_path_experiment}/{dataset}"
            try:
                raw_metrics_chi2 += open_raw_metrics(results_path, "chi2", model)
                raw_metrics_anova += open_raw_metrics(results_path, "anova", model)
                raw_metrics_forward_selection += open_raw_metrics(results_path, "forward_selection", model)
                raw_metrics_backward_elimination += open_raw_metrics(results_path, "backward_elimination", model)
            except OSError as e:
                print(f"{dataset}: {e}")

        task = "classification" if y_label == "Average Accuracy" else "regression"
        plot_path = f"results/experiment2/average_results/{task}"
        if not os.path.isdir(plot_path):
            os.makedirs(plot_path)

        algorithm_plot = plot_metrics_matplotlib(
            raw_metrics_chi2, raw_metrics_anova, raw_metrics_forward_selection, raw_metrics_backward_elimination,
            model, y_label=y_label)

        algorithm_plot.savefig(f"{plot_path}/{model}.png")
        plt.close(algorithm_plot)


def plot_average_results_experiment4(datasets: list[str], y_label="Average Accuracy"):
    results_path_experiment = "results/experiment4"
    models = ["GBM", "LR", "RF", "XGB"]
    data_types = ["categorical", "discrete", "continuous"]
    for model in models:
        for data_type in data_types:
            raw_metrics_chi2, raw_metrics_anova, raw_metrics_forward_selection, raw_metrics_backward_elimination = [], [], [], []

            for dataset in datasets:
                results_path = f"{results_path_experiment}/{dataset}/{data_type}"
                try:
                    raw_metrics_chi2 += open_raw_metrics(results_path, "chi2", model)
                    raw_metrics_anova += open_raw_metrics(results_path, "anova", model)
                    raw_metrics_forward_selection += open_raw_metrics(results_path, "forward_selection", model)
                    raw_metrics_backward_elimination += open_raw_metrics(results_path, "backward_elimination", model)
                except OSError as e:
                    print(f"{data_type}: {e}")

            task = "classification" if y_label == "Average Accuracy" else "regression"
            average_results_plot_path = f"results/experiment4/average_results/{task}"
            plot_path = f"{average_results_plot_path}/{data_type}"
            if not os.path.isdir(plot_path):
                os.makedirs(plot_path)

            algorithm_plot = plot_metrics_matplotlib(
                raw_metrics_chi2, raw_metrics_anova, raw_metrics_forward_selection, raw_metrics_backward_elimination,
                model, y_label=y_label)

            algorithm_plot.savefig(f"{plot_path}/{model}.png")
            plt.close(algorithm_plot)


def plot_experiments(experiment: str, dataset: str, y_label="Accuracy"):
    results_path = f"results/{experiment}/{dataset}"
    if experiment == "experiment4":
        data_types = ["categorical", "discrete", "continuous"]
        for data_type in data_types:
            plot_results(f"{results_path}/{data_type}", y_label)
    else:
        plot_results(results_path, y_label)


def plot_results(results_path: str, y_label="Accuracy"):
    models = ["GBM", "LR", "RF", "XGB"]

    for model in models:
        try:
            raw_metrics_chi2 = try_opening_raw_metrics(results_path, "chi2", model)
            raw_metrics_anova = try_opening_raw_metrics(results_path, "anova", model)
            raw_metrics_forward_selection = try_opening_raw_metrics(results_path, "forward_selection", model)
            raw_metrics_backward_elimination = try_opening_raw_metrics(results_path, "backward_elimination", model)

            algorithm_plot = plot_metrics_matplotlib(
                raw_metrics_chi2, raw_metrics_anova, raw_metrics_forward_selection, raw_metrics_backward_elimination,
                model, y_label=y_label)

            algorithm_plot.savefig(f"{results_path}/{model}.png")
            plt.close(algorithm_plot)
        except OSError as e:
            print(f"Failed plotting or saving: {e}")


def open_raw_metrics(results_path: str, method: str, model: str):
    with open(f"{results_path}/{method}/{model}.txt", "r", encoding="utf-8") as lines:
        return [line.strip() for line in lines]


def open_raw_runtime(runtime_path: str, method: str):
    with open(f"{runtime_path}/{method}.txt", "r", encoding="utf-8") as lines:
        return [line.strip() for line in lines]


def try_opening_raw_metrics(results_path: str, method: str, model: str):
    raw_metrics = []
    try:
        raw_metrics = open_raw_metrics(results_path, method, model)
    except OSError as e:
        print(e)
    return raw_metrics


if __name__ == '__main__':
    main()
