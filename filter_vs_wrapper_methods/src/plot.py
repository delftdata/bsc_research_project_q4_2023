from typing import Literal

import matplotlib.pyplot as plt
from plotting.plotter import plot_metrics_matplotlib


def main():
    # plot_results("experiment2", "breast_cancer")
    # plot_results("experiment2", "steel_plates_faults")
    # plot_results("experiment2", "bank_marketing")
    # plot_experiments("experiment2", "bike_sharing", y_label="Root Mean Squared Error")
    # plot_experiments("experiment4", "bank_marketing")
    # plot_experiments("experiment4", "breast_cancer")
    # plot_experiments("experiment4", "steel_plates_faults")
    # plot_experiments("experiment4", "housing_prices")
    # plot_experiments("experiment4", "bike_sharing")
    # plot_experiments("experiment4", "census_income")
    plot_experiments("experiment4", "nasa_numeric")


def plot_experiments(experiment: str, dataset: str, y_label="Accuracy"):
    if experiment == "experiment4":
        try:
            plot_results(experiment, dataset, y_label, data_type="categorical")
        except Exception as e:
            print(f"Categorical: {e}")
        try:
            plot_results(experiment, dataset, y_label, data_type="discrete")
        except Exception as e:
            print(f"Discrete: {e}")
        try:
            plot_results(experiment, dataset, y_label, data_type="continuous")
        except Exception as e:
            print(f"Continuous: {e}")
    else:
        plot_results(experiment, dataset, y_label)


def plot_results(experiment: str, dataset: str, y_label="Accuracy",
                 data_type: Literal["", "categorical", "discrete", "continuous"] = ""):
    results_path = f"results/{experiment}/{dataset}/{data_type}" if data_type else f"results/{experiment}/{dataset}"
    models = ["GBM", "LR", "RF", "XGB"]

    for model in models:
        raw_metrics_chi2 = [line.strip() for line in open(f"{results_path}/chi2/{model}.txt", "r")]
        raw_metrics_anova = [line.strip() for line in open(f"{results_path}/anova/{model}.txt", "r")]
        raw_metrics_forward_selection = [line.strip() for line in open(
            f"{results_path}/forward_selection/{model}.txt", "r")]
        raw_metrics_backward_elimination = [line.strip() for line in open(
            f"{results_path}/backward_elimination/{model}.txt", "r")]

        algorithm_plot = plot_metrics_matplotlib(
            raw_metrics_chi2, raw_metrics_anova, raw_metrics_forward_selection, raw_metrics_backward_elimination, model,
            y_label=y_label)

        algorithm_plot.savefig(f"{results_path}/{model}.png")
        plt.close(algorithm_plot)


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
