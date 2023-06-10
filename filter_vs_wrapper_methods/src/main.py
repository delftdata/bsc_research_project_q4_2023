from __future__ import annotations

import json
import os
import warnings
from multiprocessing import Pool
from typing import Literal

import pandas as pd

from evaluation.evaluator import Evaluator
from feature_selection_methods.filter import rank_features_descending_filter
from feature_selection_methods.wrapper import rank_features_descending_wrapper
from processing.splitter import (
    drop_features, drop_features_with_negative_values,
    split_categorical_discrete_continuous_features)
from reader.dataset_info import DatasetInfo
from reader.reader import Reader

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"


def main():
    """Executes the experiments of the research project `Automatic Feature Discovery: A comparative study between
    filter and wrapper feature selection techniques`.

    The arguments used by `main` can be modified by changing the values in `arguments_main.json`.

    Raises
    ------
    OSError
        If there is a mismatch between the provided `arguments_main.json` path and its actual path.
    """
    algorithm_names: list[tuple[str, str]] = [
        ("GBM", "LightGBM"), ("RF", "RandomForest"), ("LR", "LinearModel"), ("XGB", "XGBoost")]
    svm_param_grid: list[dict[str, list[str | int | float]]] = [
        {
            "C": [0.1, 100, 10, 1000],
            "gamma": [1, "scale"],
            "kernel": ["rbf", "sigmoid", "linear"]
        }]

    with open("arguments_main.json", "r", encoding="utf-8") as json_file:
        arguments = json.load(json_file)

        experiment_name: str = arguments["experiment_name"]
        print(f"Experiment name: {experiment_name}")

        preprocessing = experiment_name not in ("experiment1", "experiment3")

        imputation_strategy: Literal["mean", "median"] = arguments["imputation_strategy"]
        print(f"Imputation strategy: {imputation_strategy}")

        normalization: bool = arguments["normalization"]
        print(f"Normalization: {normalization}")

        svm: bool = arguments["evaluate_on_svm"]
        print(f"Use SVM: {svm}")

        runner = Runner(algorithm_names, svm_param_grid, experiment_name,
                        preprocessing, imputation_strategy, normalization, svm)

        dataset: str = arguments["dataset"]
        print(f"Dataset: {dataset}")

        if dataset == "small":
            runner.run_experiment_on_small_datasets_in_parallel()
        elif dataset == "big":
            runner.run_experiment_on_big_datasets_sequentially()
        else:
            runner.run_experiment_on_dataset(dataset)


class Runner:
    """
    A class used to execute the experiments and group information needed by most experiments.

    Attributes
    ----------
    algorithm_names : list[tuple[str, str]]
        List of tuples containing the algorithms on which the performance of the feature selection techniques is evaluated.
        Each tuple has the following format: ("XGB", "XGBoost").
        The first element can be an arbitrary value, and the second element reflects the name of the algorithm used by
        Autogluon, thus it should be a valid algorithm name.
    svm_param_grid : list[dict[str, list[str | int | float]]]
        List of dictionaries where the keys are SVM hyperparameter names, and the values are possible values that these
        hyperparameters can take.
    experiment_name : str
        Name of the experiment (valid experiment names: experiment1, experiment2, experiment3, experiment4).
    preprocessing : bool, optional
        Whether to perform preprocessing steps (default: True).
    imputation_strategy : Literal["mean", "median"], optional
        Name of the imputation strategy used to handle missing values (default: "mean").
    normalization : bool, optional
        Whether to normalize the datasets on which ANOVA and the wrapper methods operate on (default: True).
    svm : bool, optional
        Whether to use the sklearn SVM variants (Linear SVC and Linear SVR) in the evaluation process rather than the
        Autogluon algorithms (default: False).
    reader_dictionary : dict[str, Callable]
        Dictionary that binds the dataset name to a method used to read the specific dataset.
    """

    def __init__(self, algorithm_names: list[tuple[str, str]], svm_param_grid: list[dict[str, list[str | int | float]]],
                 experiment_name: str, preprocessing=True, imputation_strategy: Literal["mean", "median"] = "mean",
                 normalization=True, svm=False):
        """
        Parameters
        ----------
        algorithm_names : list[tuple[str, str]]
            List of tuples containing the algorithms on which the performance of the feature selection techniques is evaluated.
            Each tuple has the following format: ("XGB", "XGBoost").
            The first element can be an arbitrary value, and the second element reflects the name of the algorithm used by
            Autogluon, thus it should be a valid algorithm name.
        svm_param_grid : list[dict[str, list[str | int | float]]]
            List of dictionaries where the keys are SVM hyperparameter names, and the values are possible values that these
            hyperparameters can take.
        experiment_name : str
            Name of the experiment (valid experiment names: experiment1, experiment2, experiment3, experiment4).
        preprocessing : bool, optional
            Whether to perform preprocessing steps (default: True).
        imputation_strategy : Literal["mean", "median"], optional
            Name of the imputation strategy used to handle missing values (default: "mean").
        normalization : bool, optional
            Whether to normalize the datasets on which ANOVA and the wrapper methods operate on (default: True).
        svm : bool, optional
            Whether to use the sklearn SVM variants (Linear SVC and Linear SVR) in the evaluation process rather than the
            Autogluon algorithms (default: False).
        """
        self.algorithm_names = algorithm_names
        self.svm_param_grid = svm_param_grid
        self.experiment_name = experiment_name
        self.preprocessing = preprocessing
        self.imputation_strategy: Literal["mean", "median"] = imputation_strategy
        self.normalization = normalization
        self.svm = svm
        reader = Reader(experiment_name, imputation_strategy)
        self.reader_dictionary = {
            "bank_marketing": reader.read_bank_marketing,
            "breast_cancer": reader.read_breast_cancer,
            "steel_plates_faults": reader.read_steel_plates_faults,
            "housing_prices": reader.read_housing_prices,
            "bike_sharing": reader.read_bike_sharing,
            "census_income": reader.read_census_income,
            "connect_4": reader.read_connect_4,
            "arrhythmia": reader.read_arrhythmia,
            "crop": reader.read_crop,
            "character_font_images": reader.read_character_font_images,
            "internet_ads": reader.read_internet_ads,
            "nasa_numeric": reader.read_nasa_numeric,
        }

    def run_experiment_on_dataset(self, dataset: str):
        """Runs the experiment on a specific dataset.

        Parameters
        ----------
        dataset : str
            The name of the dataset to run the experiment on.

        Raises
        ------
        KeyError
            If the dataset name is not found in the reader_dictionary.
        """
        df, dataset_info = self.reader_dictionary[dataset]()
        if self.experiment_name == "experiment3":
            self.run_experiment3(df, dataset_info)
        elif self.experiment_name == "experiment4":
            self.run_experiment4(df, dataset_info)
        else:
            self.evaluate_feature_selection(df, dataset_info)

    def run_experiment_on_small_datasets_in_parallel(self):
        """Runs the experiment on small datasets in parallel using multiprocessing.

        Notes
        -----
        - The small datasets are the following:
            - "bank_marketing"
            - "bike_sharing"
            - "breast_cancer"
            - "steel_plates_faults"
            - "census_income"
            - "housing_prices"
            - "nasa_numeric"
            - "connect_4"
        """
        with Pool() as pool:
            small_datasets = ["bank_marketing", "bike_sharing", "breast_cancer", "steel_plates_faults",
                              "census_income", "housing_prices", "nasa_numeric", "connect_4"]
            pool.map(self.run_experiment_on_dataset, small_datasets)

    def run_experiment_on_big_datasets_sequentially(self):
        """Runs the experiment on big datasets sequentially.

        Notes
        -----
        - The big datasets are the following:
            - "arrhythmia"
            - "crop"
            - "character_font_images"
            - "internet_ads"
        """
        big_datasets = ["arrhythmia", "crop", "character_font_images", "internet_ads"]
        for big_dataset in big_datasets:
            self.run_experiment_on_dataset(dataset=big_dataset)

    def run_experiment3(self, df: pd.DataFrame, dataset_info: DatasetInfo, min_columns=2):
        """Runs experiment 3 on the given DataFrame using Chi-Squared, ANOVA, Forward Selection and Backward Elimination
        feature selection methods.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to run the experiment on.
        dataset_info : DatasetInfo
            The information about the dataset needed for the experiment.
        min_columns : int, optional
            The minimum number of columns required for a feature selection method to select (default: 2).
            Each method will only be evaluated if the number of selected columns in the DataFrame exceeds this threshold.
            This is due to the fact that Autogluon usually cannot fit the models on data with 2 columns or less.
        """
        df_chi2 = df.copy()
        df_anova = df.copy()
        df_forward_selection = df.copy()
        df_backward_elimination = df.copy()

        df_chi2 = drop_features(df_chi2, dataset_info.target_label, "string")
        df_chi2 = drop_features_with_negative_values(df_chi2, dataset_info.target_label)

        df_anova = drop_features(df_anova, dataset_info.target_label, "string")
        df_anova = drop_features(df_anova, dataset_info.target_label, "int64")

        df_forward_selection = drop_features(df_forward_selection, dataset_info.target_label, "string")
        df_backward_elimination = drop_features(df_backward_elimination, dataset_info.target_label, "string")

        if df_chi2.columns.size > min_columns:
            self.evaluate_feature_selection(df_chi2, dataset_info, methods=("chi2"))
        if df_anova.columns.size > min_columns:
            self.evaluate_feature_selection(df_anova, dataset_info, methods=("anova"))
        if df_forward_selection.columns.size > min_columns:
            self.evaluate_feature_selection(df_forward_selection, dataset_info, methods=("forward_selection"))
        if df_backward_elimination.columns.size > min_columns:
            self.evaluate_feature_selection(df_backward_elimination, dataset_info, methods=("backward_elimination"))

    def run_experiment4(self, df: pd.DataFrame, dataset_info: DatasetInfo, min_columns=2):
        """Runs experiment 4 on the given DataFrame using Chi-Squared, ANOVA, Forward Selection and Backward Elimination
        feature selection methods.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to run the experiment on.
        dataset_info : DatasetInfo
            The information about the dataset needed for the experiment.
        min_columns : int, optional
            The minimum number of columns required for a feature selection method to select (default: 2).
            Each method will only be evaluated if the number of selected columns in the DataFrame exceeds this threshold.
            This is due to the fact that Autogluon usually cannot fit the models on data with 2 columns or less.
        """
        df_categorical, df_discrete, df_continuous = \
            split_categorical_discrete_continuous_features(df, target_label=dataset_info.target_label)
        dataset_info_categorical = DatasetInfo(
            dataset_info.dataset_file, dataset_info.target_label, f"{dataset_info.results_path}/categorical",
            eval_metric=dataset_info.eval_metric)
        dataset_info_discrete = DatasetInfo(
            dataset_info.dataset_file, dataset_info.target_label, f"{dataset_info.results_path}/discrete",
            eval_metric=dataset_info.eval_metric)
        dataset_info_continuous = DatasetInfo(
            dataset_info.dataset_file, dataset_info.target_label, f"{dataset_info.results_path}/continuous",
            eval_metric=dataset_info.eval_metric)

        if df_categorical.columns.size > min_columns:
            self.evaluate_feature_selection(df_categorical, dataset_info_categorical)
        if df_discrete.columns.size > min_columns:
            self.evaluate_feature_selection(df_discrete, dataset_info_discrete)
        if df_continuous.columns.size > min_columns:
            self.evaluate_feature_selection(df_continuous, dataset_info_continuous)

    def evaluate_feature_selection(
            self, df: pd.DataFrame, dataset_info: DatasetInfo,
            methods=("chi2", "anova", "forward_selection", "backward_elimination")):
        """Performs feature selection on the given DataFrame using Chi-Squared, ANOVA, Forward Selection, Backward Elimination
        and evaluates their performance.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame containing the dataset.
        dataset_info : DatasetInfo
            An instance of the DatasetInfo class representing the dataset information.
        methods : tuple, optional
            The feature selection methods to apply (default: ("chi2", "anova", "forward_selection", "backward_elimination")).
        """

        selected_features_path = f"{dataset_info.results_path}/selected_features"

        for method in methods:
            if not os.path.isfile(f"{selected_features_path}/{method}.txt"):
                filter_methods: set[Literal["chi2", "anova"]] = set(["chi2", "anova"])

                if method in filter_methods:
                    sorted_features, runtime = rank_features_descending_filter(
                        df, method, dataset_info.target_label, self.preprocessing, self.normalization)
                else:
                    sorted_features, runtime = rank_features_descending_wrapper(
                        df, method, dataset_info.target_label, dataset_info.eval_metric,
                        self.preprocessing, self.normalization)

                write_runtime(dataset_info, runtime, method)
                write_selected_features(dataset_info, sorted_features, method)
            print(f"Finished feature selection, {method}.")

        evaluator = Evaluator(df, dataset_info.target_label, "root_mean_squared_error"
                              if dataset_info.eval_metric == "neg_root_mean_squared_error" else dataset_info.eval_metric,
                              self.algorithm_names, self.svm_param_grid)

        for method in methods:
            try:
                with open(f"{selected_features_path}/{method}.txt", "r", encoding="utf-8") as lines:
                    sorted_features = [line.strip() for line in lines]
                    performance = evaluator.perform_experiments(sorted_features, svm=self.svm)
                    print(f"Autogluon finished evaluating the features selected by: {method}.")
                    write_performance(dataset_info, performance, method)
            except Exception as error:
                print(f"Autogluon could not evaluate method -{method}-, {error}.")


def write_selected_features(dataset_info: DatasetInfo, selected_features: list[str], method: str):
    """Writes the selected features to a file for a specific feature selection method and dataset.

    Parameters
    ----------
    dataset_info : DatasetInfo
        An instance of the DatasetInfo class representing the dataset information.
    selected_features : list[str]
        The list of selected features.
    method : str
        The feature selection method.
    """
    for selected_feature in selected_features:
        write_to_file(f"{dataset_info.results_path}/selected_features", f"{method}.txt", selected_feature)


def write_runtime(dataset_info: DatasetInfo, runtime: float, method: str):
    """Writes the runtime of a feature selection method to a file for a specific dataset.

    Parameters
    ----------
    dataset_info : DatasetInfo
        An instance of the DatasetInfo class representing the dataset information.
    runtime : float
        The runtime of the feature selection method.
    method : str
        The feature selection method.
    """
    write_to_file(f"{dataset_info.results_path}/runtime", f"{method}.txt", str(runtime))


def write_performance(dataset_info: DatasetInfo, performance: dict[str, list[float]], method):
    """Writes the performance results of a feature selection method for different algorithms to files.

    Parameters
    ----------
    dataset_info : DatasetInfo
        An instance of the DatasetInfo class representing the dataset information.
    performance : dict[str, list[float]]
        A dictionary mapping algorithm names to a list of performance values.
    method : str
        The feature selection method.
    """
    for (algorithm, performance_algorithm) in performance.items():
        content = ",".join([str(x) for x in performance_algorithm])
        write_to_file(f"{dataset_info.results_path}/{method}", f"{algorithm}.txt", content)


def write_to_file(path: str, results_file_name: str, content: str, mode="a+"):
    """Writes the provided content to a file specified by the path and filename.

    Parameters
    ----------
    path : str
        The path to the directory where the file should be stored.
    results_file_name : str
        The name of the file to write the content to.
    content : str
        The content to write to the file.
    mode : str, optional
        The mode in which the file should be opened (default: "a+").
        Supported modes: "r", "w", "a", "r+", "w+", "a+", etc.
    """
    path_to_file = f"{path}/{results_file_name}"
    try:
        if not os.path.isdir(path):
            os.makedirs(path)
        with open(path_to_file, mode=mode, encoding="utf-8") as file:
            file.write(content + "\n")
    except OSError as error:
        print(f"An error occurred: {error}")


if __name__ == "__main__":
    main()
