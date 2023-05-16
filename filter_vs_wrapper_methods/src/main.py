import os
import warnings

import pandas as pd
from methods.filter import Filter
from methods.wrapper import Wrapper
from ML_models.models import AutogluonModel
from sklearn.calibration import LinearSVC

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"


def perform_feature_selection_on_steel_plates_faults_dataset(
        df: pd.DataFrame, target_index=0, selected_features_size=0.6):
    chi2, anova, forward_selection, backward_elimination = None, None, None, None

    try:
        chi2 = Filter.perform_feature_selection(
            df, target_index, "chi2", selected_features_size=selected_features_size)
    except Exception as e:
        print("Chi2:", e)

    try:
        anova = Filter.perform_feature_selection(
            df, target_index, "anova", selected_features_size=selected_features_size)
    except Exception as e:
        print("Anova:", e)

    try:
        forward_selection = Wrapper.perform_feature_selection(
            df, target_index, LinearSVC(), direction="forward", selected_features_size=selected_features_size)
    except Exception as e:
        print("Forward Selection: ", e)

    try:
        backward_elimination = Wrapper.perform_feature_selection(
            df, target_index, LinearSVC(), direction="backward", selected_features_size=selected_features_size)
    except Exception as e:
        print("Backward Elimination: ", e)

    return chi2, anova, forward_selection, backward_elimination


def write_to_file(path: str, file_name: str, content: str):
    path_to_file = f"{path}/{file_name}"
    try:
        if not os.path.isdir(path):
            os.makedirs(path)
        with open(path_to_file, "a+") as file:
            file.write(content + "\n")
        print(f"Updated -{path_to_file}- with content -{content}-")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    hyperparameters = [
        {
            "LR": {}
        },
        {
            "XGB": {},
        },
        {
            "GBM": {},
        },
        {
            "RF": {},
        }]
    hyperparameters_names = [
        "LR", "XGB",
        "GBM", "RF"]

    param_grid = [
        {
            "C": [0.1, 10, 1000],
            "gamma": [1, "scale"],
            "kernel": ["rbf", "sigmoid", "linear"]
        },
        {
            "C": [0.1, 10, 1000],
            "gamma": [1, "scale"],
            "kernel": ["poly"],
            "degree": [6, 7, 8, 9]
        }
    ]

    file_dataset = "data/steel_plates_faults/steel_plates_faults.csv"
    df_plates = pd.read_csv(file_dataset)
    df_plates = df_plates.iloc[:, [df_plates.shape[1] - 1] + [i for i in range(df_plates.shape[1] - 1)]]

    chi2, anova, forward_selection, backward_elimination = perform_feature_selection_on_steel_plates_faults_dataset(
        df_plates)

    data_frames: list[tuple[str, pd.DataFrame]] = [
        ("chi2", chi2),
        ("anova", anova),
        ("forward_selection", forward_selection),
        ("backward_elimination", backward_elimination),
        ("original", df_plates)]

    selected_data_frames = [(method_name, df) for (method_name, df) in data_frames if df is not None]
    for (method_name, df) in selected_data_frames:
        print(method_name)
        print(df.head())

    results_path = "results/breast_cancer"
    file_name = "metrics.txt"

    for (method_name, df) in selected_data_frames:
        for i, hyperparameter in enumerate(hyperparameters):
            autogluon_model = AutogluonModel(label=df.columns[0],
                                             problem_type="binary", data_preprocessing=False,
                                             hyperparameters=hyperparameter)
            autogluon_model.fit(df)

            write_to_file(f"{results_path}/{method_name}/{hyperparameters_names[i]}", file_name,
                          str(autogluon_model.evaluate()))

        # svm_model = SVMModel(label=df.columns[0], problem_type="binary",
        #                      data_preprocessing=False)
        # svm_model.grid_search(df, param_grid=param_grid)

        # write_to_file(f"{results_path}/{method_name}/SVM", file_name,
        #               str(svm_model.evaluate()))
