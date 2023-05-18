import os
import warnings
from typing import Literal

import pandas as pd
from utility.evaluator import Evaluator
from utility.plotter import Plotter

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"


def main():
    algorithms_names: list[tuple[str, str]] = [
        ("GBM", "LightGBM"), ("RF", "RandomForest"), ("LR", "LinearModel"), ("XGB", "XGBoost")]
    # param_grid = [{"C": [0.1, 10, 1000], "gamma": [1, "scale"], "kernel": ["rbf", "sigmoid", "linear"]},
    #               {"C": [0.1, 10, 1000], "gamma": [1, "scale"], "kernel": ["poly"], "degree": [6, 7, 8, 9]}]

    dataset_files = ["data/steel_plates_faults/steel_plates_faults.csv"]
    target_labels = ["Class"]
    results_paths = ["results/steel_plates_faults"]
    eval_metrics: list[Literal["accuracy", "neg_mean_squared_error"]] = ["accuracy"]
    scoring_methods: list[Literal["Accuracy", "Mean Squared Error"]] = ["Accuracy"]

    for i in range(len(dataset_files)):
        dataset_file, target_label, results_path, eval_metric, scoring = \
            dataset_files[i], target_labels[i],  results_paths[i], eval_metrics[i], scoring_methods[i]

        df = pd.read_csv(dataset_file, low_memory=False)

        evaluator = Evaluator(df=df, target_label=target_label, scoring=eval_metric, algorithms_names=algorithms_names)
        performances = evaluator.perform_experiments()

        for (algorithm, performance_algorithm) in performances.items():
            plot_results(path=results_path, performance_algorithm=performance_algorithm,
                         scoring=scoring, algorithm=algorithm)
            write_to_file(path=results_path, results_file_name=f"{algorithm}.txt", content=str(performance_algorithm))


def plot_results(
        path: str, performance_algorithm: dict[str, list[float]],
        scoring: Literal['Accuracy', 'Mean Squared Error'],
        algorithm: str):
    try:
        if not os.path.isdir(path):
            os.makedirs(path)
        algorithm_plot = Plotter.plot_metric_sea_born(performance=performance_algorithm, scoring=scoring)
        figure = algorithm_plot.get_figure()
        figure.savefig(f"{path}/{algorithm}.png")
        figure.clf()
    except Exception as e:
        print(f"An error occurred: {e}")


def write_to_file(path: str, results_file_name: str, content: str):
    path_to_file = f"{path}/{results_file_name}"
    try:
        if not os.path.isdir(path):
            os.makedirs(path)
        with open(path_to_file, "a+") as file:
            file.write(content + "\n")
        print(f"Updated -{path_to_file}- with content -{content}-")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
