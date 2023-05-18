import os
import warnings

import pandas as pd
from utility.evaluator import Evaluator

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"


def main():
    algorithms_names: list[tuple[str, str]] = [
        ("GBM", "LightGBM"), ("RF", "RandomForest"), ("LR", "LinearModel"), ("XGB", "XGBoost")]
    # param_grid = [{"C": [0.1, 10, 1000], "gamma": [1, "scale"], "kernel": ["rbf", "sigmoid", "linear"]},
    #               {"C": [0.1, 10, 1000], "gamma": [1, "scale"], "kernel": ["poly"], "degree": [6, 7, 8, 9]}]
    results_file_name = "metrics.txt"

    dataset_files = ["data/steel_plates_faults/steel_plates_faults.csv"]
    target_labels = ["Class"]
    results_paths = ["results/steel_plates_faults"]
    eval_metrics = ["accuracy"]

    for i in range(len(dataset_files)):
        dataset_file, target_label, results_path, eval_metric = \
            dataset_files[i], target_labels[i],  results_paths[i], eval_metrics[i]

        df = pd.read_csv(dataset_file, low_memory=False)
        selected_data_frames = Evaluator.perform_feature_selection_methods(
            df, target_label=target_label, scoring=eval_metric)

        for (method_name, df) in selected_data_frames:
            print(method_name)
            print(df.head())

        results = Evaluator.evaluate_models(
            selected_data_frames=selected_data_frames, target_label=target_label, algorithms_names=algorithms_names,
            eval_metric=eval_metric)

        for (method_name, algorithm, performance) in results:
            write_to_file(f"{results_path}/{method_name}/{algorithm}", results_file_name, performance)


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
