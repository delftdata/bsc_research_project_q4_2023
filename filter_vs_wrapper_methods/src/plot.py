import os

import matplotlib.pyplot as plt
from plotting.plotter import plot_metrics_matplotlib


def main():
    # plot_results("experiment2", "breast_cancer")
    # plot_results("experiment2", "steel_plates_faults")
    # plot_results("experiment2", "bank_marketing")
    plot_experiments("experiment3", "bank_marketing")
    plot_experiments("experiment3", "bike_sharing", y_label="Root Mean Squared Error")
    plot_experiments("experiment3", "breast_cancer")
    plot_experiments("experiment3", "census_income")
    plot_experiments("experiment3", "connect-4")
    plot_experiments("experiment3", "housing_prices", y_label="Root Mean Squared Error")
    plot_experiments("experiment3", "steel_plates_faults")
    plot_experiments("experiment3", "arrhythmia")
    # plot_experiments("experiment3", "nasa_numeric", y_label="Root Mean Squared Error")
    # classification_datasets = ["arrhythmia", "bank_marketing", "breast_cancer", "census_income", "steel_plates_faults"]
    # regression_datasets = ["housing_prices", "bike_sharing", "nasa_numeric"]
    # y_label_regression = "Average Root Mean Squared Error"
    # # plot_average_results_experiment2(classification_datasets)
    # # plot_average_results_experiment2(regression_datasets, y_label=y_label_regression)
    # plot_average_results_experiment4(classification_datasets)
    # plot_average_results_experiment4(regression_datasets, y_label=y_label_regression)


def plot_average_results_experiment2(datasets: list[str], y_label="Average Accuracy", dpi=2500):
    results_path_experiment = "results/experiment2"
    models = ["GBM", "LR", "RF", "XGB"]
    for model in models:
        raw_metrics_chi2, raw_metrics_anova, raw_metrics_forward_selection, raw_metrics_backward_elimination = [], [], [], []
        for dataset in datasets:
            results_path = f"{results_path_experiment}/{dataset}"
            try:
                with open(f"{results_path}/chi2/{model}.txt", "r") as lines:
                    raw_metrics_chi2 += [line.strip() for line in lines]
                with open(f"{results_path}/anova/{model}.txt", "r") as lines:
                    raw_metrics_anova += [line.strip() for line in lines]
                with open(f"{results_path}/forward_selection/{model}.txt", "r") as lines:
                    raw_metrics_forward_selection += [line.strip() for line in lines]
                with open(f"{results_path}/backward_elimination/{model}.txt", "r") as lines:
                    raw_metrics_backward_elimination += [line.strip() for line in lines]
            except Exception as e:
                print(f"{dataset}: {e}")

        task = "classification" if y_label == "Average Accuracy" else "regression"
        plot_path = f"results/experiment2/average_results/{task}"
        if not os.path.isdir(plot_path):
            os.makedirs(plot_path)

        algorithm_plot = plot_metrics_matplotlib(
            raw_metrics_chi2, raw_metrics_anova, raw_metrics_forward_selection, raw_metrics_backward_elimination,
            model, y_label=y_label)

        algorithm_plot.savefig(f"{plot_path}/{model}.png", dpi=dpi)
        plt.close(algorithm_plot)


def plot_average_results_experiment4(datasets: list[str], y_label="Average Accuracy", dpi=2500):
    results_path_experiment = "results/experiment4"
    models = ["GBM", "LR", "RF", "XGB"]
    data_types = ["categorical", "discrete", "continuous"]
    for model in models:
        for data_type in data_types:
            raw_metrics_chi2, raw_metrics_anova, raw_metrics_forward_selection, raw_metrics_backward_elimination = [], [], [], []
            for dataset in datasets:
                results_path = f"{results_path_experiment}/{dataset}/{data_type}"
                try:
                    with open(f"{results_path}/chi2/{model}.txt", "r") as lines:
                        raw_metrics_chi2 += [line.strip() for line in lines]
                    with open(f"{results_path}/anova/{model}.txt", "r") as lines:
                        raw_metrics_anova += [line.strip() for line in lines]
                    with open(f"{results_path}/forward_selection/{model}.txt", "r") as lines:
                        raw_metrics_forward_selection += [line.strip() for line in lines]
                    with open(f"{results_path}/backward_elimination/{model}.txt", "r") as lines:
                        raw_metrics_backward_elimination += [line.strip() for line in lines]
                except Exception as e:
                    print(f"{data_type}: {e}")

            task = "classification" if y_label == "Average Accuracy" else "regression"
            average_results_plot_path = f"results/experiment4/average_results/{task}"
            plot_path = f"{average_results_plot_path}/{data_type}"
            if not os.path.isdir(plot_path):
                os.makedirs(plot_path)

            algorithm_plot = plot_metrics_matplotlib(
                raw_metrics_chi2, raw_metrics_anova, raw_metrics_forward_selection, raw_metrics_backward_elimination,
                model, y_label=y_label)

            algorithm_plot.savefig(f"{plot_path}/{model}.png", dpi=dpi)
            plt.close(algorithm_plot)


def plot_experiments(experiment: str, dataset: str, y_label="Accuracy"):
    results_path = f"results/{experiment}/{dataset}"
    if experiment == "experiment4":
        try:
            plot_results(f"{results_path}/categorical", y_label)
        except Exception as e:
            print(f"Categorical: {e}")
        try:
            plot_results(f"{results_path}/discrete", y_label)
        except Exception as e:
            print(f"Discrete: {e}")
        try:
            plot_results(f"{results_path}/continuous", y_label)
        except Exception as e:
            print(f"Continuous: {e}")
    else:
        plot_results(results_path, y_label)


def plot_results(results_path: str, y_label="Accuracy", dpi=2500):
    models = ["GBM", "LR", "RF", "XGB"]

    for model in models:
        try:
            raw_metrics_chi2 = try_reading(results_path, "chi2", model)
            raw_metrics_anova = try_reading(results_path, "anova", model)
            raw_metrics_forward_selection = try_reading(results_path, "forward_selection", model)
            raw_metrics_backward_elimination = try_reading(results_path, "backward_elimination", model)

            algorithm_plot = plot_metrics_matplotlib(
                raw_metrics_chi2, raw_metrics_anova, raw_metrics_forward_selection, raw_metrics_backward_elimination,
                model, y_label=y_label)

            algorithm_plot.savefig(f"{results_path}/{model}.png", dpi=dpi)
            plt.close(algorithm_plot)
        except Exception as e:
            print(f"Failed plotting or reading: {e}")


def try_reading(results_path: str, method: str, model: str):
    raw_metrics = []
    try:
        raw_metrics = [line.strip() for line in open(f"{results_path}/{method}/{model}.txt", "r")]
    except Exception as e:
        print(e)
    return raw_metrics


# def plot_results(path: str, performance_algorithm: dict[str, list[float]], scoring, algorithm: str):
#     try:
#         if not os.path.isdir(path):
#             os.makedirs(path)
#         algorithm_plot = plot_metrics_sea_born(performance=performance_algorithm, scoring=scoring)
#         figure = algorithm_plot.get_figure()
#         figure.savefig(f"{path}/{algorithm}.png")
#         figure.clf()
#     except Exception as e:
#         print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()
