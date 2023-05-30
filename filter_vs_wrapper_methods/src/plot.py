import matplotlib.pyplot as plt
from plotting.plotter import Plotter


def main():
    # plot_results("experiment2", "breast_cancer")
    # plot_results("experiment2", "steel_plates_faults")
    # plot_results("experiment2", "bank_marketing")
    plot_results("experiment2", "bike_sharing", y_label="Root Mean Squared Error")


def plot_results(experiment: str, dataset: str, y_label="Accuracy"):
    results_path = f"results/{experiment}/{dataset}"
    models = ["GBM", "LR", "RF", "XGB"]

    for model in models:
        raw_metrics_chi2 = [line.strip() for line in open(f"{results_path}/chi2/{model}.txt", "r")]
        raw_metrics_anova = [line.strip() for line in open(f"{results_path}/anova/{model}.txt", "r")]
        raw_metrics_forward_selection = [line.strip() for line in open(
            f"{results_path}/forward_selection/{model}.txt", "r")]
        raw_metrics_backward_elimination = [line.strip() for line in open(
            f"{results_path}/backward_elimination/{model}.txt", "r")]

        algorithm_plot = Plotter.plot_metric_matplotlib(
            raw_metrics_chi2, raw_metrics_anova, raw_metrics_forward_selection, raw_metrics_backward_elimination, model,
            y_label=y_label)

        algorithm_plot.savefig(f"{results_path}/{model}.png")
        plt.close(algorithm_plot)


# def plot_results(path: str, performance_algorithm: dict[str, list[float]], scoring, algorithm: str):
#     try:
#         if not os.path.isdir(path):
#             os.makedirs(path)
#         algorithm_plot = Plotter.plot_metric_sea_born(performance=performance_algorithm, scoring=scoring)
#         figure = algorithm_plot.get_figure()
#         figure.savefig(f"{path}/{algorithm}.png")
#         figure.clf()
#     except Exception as e:
#         print(f"An error occurred: {e}")


if __name__ == '__main__':
    main()
