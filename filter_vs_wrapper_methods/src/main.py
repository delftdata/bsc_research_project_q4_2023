from __future__ import annotations

import json
import os
import warnings
from typing import Literal

import pandas as pd
from autogluon.features.generators import AutoMLPipelineFeatureGenerator
from autogluon.tabular import TabularDataset, TabularPredictor
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC, SVR

from evaluator.evaluator import Evaluator
from feature_selection_methods.filter import rank_features_descending_filter
from feature_selection_methods.wrapper import rank_features_descending_wrapper
from processing.preprocessing import discretize_columns_ordinal_encoder
from processing.splitter import (
    drop_features, drop_features_with_negative_values,
    split_categorical_discrete_continuous_features, split_input_target)
from reader.dataset_info import DatasetInfo
from reader.reader import Reader
from writer.writer import Writer

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

        runner.run_experiment_on_dataset(dataset)


class Runner:
    """
    A class used to execute the experiments and group information needed by most experiments.

    Attributes
    ----------
    algorithm_names : list[tuple[str, str]]
        List of tuples containing the algorithms on which the performance of the feature selection
        techniques is evaluated.
        Each tuple has the following format: ("XGB", "XGBoost").
        The first element can be an arbitrary value, and the second element reflects the name of the algorithm
        used by Autogluon, thus it should be a valid algorithm name.
    svm_param_grid : list[dict[str, list[str | int | float]]]
        List of dictionaries where the keys are SVM hyperparameter names, and the values are possible values
        that these hyperparameters can take.
    experiment_name : str
        Name of the experiment (valid experiment names: experiment1, experiment2, experiment3, experiment4).
    preprocessing : bool, optional
        Whether to perform preprocessing steps (default: True).
    imputation_strategy : Literal["mean", "median"], optional
        Name of the imputation strategy used to handle missing values (default: "mean").
    normalization : bool, optional
        Whether to normalize the datasets on which ANOVA and the wrapper methods operate on (default: True).
    svm : bool, optional
        Whether to use the sklearn SVM variants (SVC and SVR) in the evaluation process
        rather than the Autogluon algorithms (default: False).
    reader_dictionary : dict[str, Callable]
        Dictionary that binds the dataset name to a method used to read the specific dataset.
    big_datasets : list[str]
        List of datasets too big to be evaluated by wrapper methods in less than a week.
    """

    def __init__(self, algorithm_names: list[tuple[str, str]], svm_param_grid: list[dict[str, list[str | int | float]]],
                 experiment_name: str, preprocessing: bool, imputation_strategy: Literal["mean", "median"],
                 normalization: bool, svm: bool):
        """
        Parameters
        ----------
        algorithm_names : list[tuple[str, str]]
            List of tuples containing the algorithms on which the performance of the feature selection
            techniques is evaluated.
            Each tuple has the following format: ("XGB", "XGBoost").
            The first element can be an arbitrary value, and the second element reflects the name of the algorithm
            used by Autogluon, thus it should be a valid algorithm name.
        svm_param_grid : list[dict[str, list[str | int | float]]]
            List of dictionaries where the keys are SVM hyperparameter names, and the values are possible values
            that these hyperparameters can take.
        experiment_name : str
            Name of the experiment (valid experiment names: experiment1, experiment2, experiment3, experiment4).
        preprocessing : bool, optional
            Whether to perform preprocessing steps (default: True).
        imputation_strategy : Literal["mean", "median"], optional
            Name of the imputation strategy used to handle missing values (default: "mean").
        normalization : bool, optional
            Whether to normalize the datasets on which ANOVA and the wrapper methods operate on (default: True).
        svm : bool, optional
            Whether to use the sklearn SVM variants (SVC and SVR) in the evaluation process
            rather than the Autogluon algorithms (default: False).
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
            "arrhythmia": reader.read_arrhythmia,
            "crop": reader.read_crop,
            "character_font_images": reader.read_character_font_images,
            "internet_advertisements": reader.read_internet_advertisements,
            "nasa_numeric": reader.read_nasa_numeric,
        }
        self.big_datasets = ["internet_advertisements", "arrhythmia", "crop", "character_font_images"]

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
        df_categorical, df_discrete, df_continuous = split_categorical_discrete_continuous_features(
            df, target_label=dataset_info.target_label)
        dataset_info_categorical = DatasetInfo(
            dataset_info.dataset_name, dataset_info.dataset_path, dataset_info.target_label,
            f"{dataset_info.results_path}/categorical", eval_metric=dataset_info.eval_metric)
        dataset_info_discrete = DatasetInfo(
            dataset_info.dataset_name, dataset_info.dataset_path, dataset_info.target_label,
            f"{dataset_info.results_path}/discrete", eval_metric=dataset_info.eval_metric)
        dataset_info_continuous = DatasetInfo(
            dataset_info.dataset_name, dataset_info.dataset_path, dataset_info.target_label,
            f"{dataset_info.results_path}/continuous", eval_metric=dataset_info.eval_metric)

        if df_categorical.columns.size > min_columns:
            self.evaluate_feature_selection(df_categorical, dataset_info_categorical)
        if df_discrete.columns.size > min_columns:
            self.evaluate_feature_selection(df_discrete, dataset_info_discrete)
        if df_continuous.columns.size > min_columns:
            self.evaluate_feature_selection(df_continuous, dataset_info_continuous)

    def evaluate_feature_selection(
            self, df: pd.DataFrame, dataset_info: DatasetInfo,
            methods=("chi2", "anova", "forward_selection", "backward_elimination")):
        """Performs feature selection on the given DataFrame using Chi-Squared, ANOVA, Forward Selection,
        Backward Elimination and evaluates their performance.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame containing the dataset.
        dataset_info : DatasetInfo
            An instance of the DatasetInfo class representing the dataset information.
        methods : tuple, optional
            The feature selection methods to apply (default:
            ("chi2", "anova", "forward_selection", "backward_elimination")).
        """
        results_path = self.get_results_path(dataset_info.results_path)
        selected_features_path = f"{results_path}/selected_features"
        filter_methods = ("chi2", "anova")

        if dataset_info.dataset_name in self.big_datasets and len(methods) == 4:
            methods = filter_methods
        elif dataset_info.dataset_name in self.big_datasets and len(methods) == 1 and methods not in filter_methods:
            return

        for method in methods:
            if not os.path.isfile(f"{selected_features_path}/{method}.txt"):
                if method in filter_methods:
                    sorted_features, runtime = rank_features_descending_filter(
                        df, method, dataset_info.target_label, self.preprocessing, self.normalization)
                else:
                    sorted_features, runtime = rank_features_descending_wrapper(
                        df, method, dataset_info.target_label, dataset_info.eval_metric,
                        self.preprocessing, self.normalization)

                Writer.write_runtime(results_path, runtime, method)
                Writer.write_selected_features(results_path, sorted_features, method)
            print(f"Finished feature selection, {method}.")

        scoring = "root_mean_squared_error" if dataset_info.eval_metric == "neg_root_mean_squared_error" else dataset_info.eval_metric

        if self.svm:
            estimator = SVC() if scoring == "accuracy" else SVR()
            hyperparameters = self.get_hyperparameters_svm(estimator, df, dataset_info)
        else:
            hyperparameters = self.get_hyperparameters(df, dataset_info, scoring)

        evaluator = Evaluator(df, dataset_info.target_label, scoring, self.algorithm_names, hyperparameters)

        for method in methods:
            try:
                with open(f"{selected_features_path}/{method}.txt", "r", encoding="utf-8") as lines:
                    sorted_features = [line.strip() for line in lines]
                    performance = evaluator.perform_experiments(sorted_features, svm=self.svm)
                    print(f"Finished evaluating the features selected by: {method}.")
                    Writer.write_performance(results_path, performance, method)
            except Exception as error:
                print(f"-{method}-, {error}.")

    def get_results_path(self, results_path: str) -> str:
        """Returns the results path based on the configuration of the experiment.

        Parameters
        ----------
        results_path : str
            The base path to the results directory.

        Returns
        -------
        str
            The updated results path based on the experiment configuration.
        """
        if not self.normalization and self.imputation_strategy == "median":
            results_path = f"{results_path}/no_normalization_median"
        elif not self.normalization:
            results_path = f"{results_path}/no_normalization"
        elif self.imputation_strategy == "median":
            results_path = f"{results_path}/median"
        return results_path

    def get_hyperparameters(self, df: pd.DataFrame, dataset_info: DatasetInfo, scoring: str) -> dict[str, dict]:
        """Retrieves or performs hyperparameter tuning for multiple algorithms on the given dataset.

        Parameters
        ----------
        df : pd.DataFrame
            The dataset to perform hyperparameter tuning on.
        dataset_info : DatasetInfo
            Information about the dataset, including the target label and evaluation metric.
        scoring : str
            The evaluation metric used for hyperparameter tuning.

        Returns
        -------
        dict[str, dict]
            A dictionary containing the hyperparameters for each algorithm.
        """
        hyperparameters_path = f"results/hyperparameters/{dataset_info.dataset_name}"
        hyperparameters: dict[str, dict] = {}

        for (algorithm, algorithm_name) in self.algorithm_names:
            if not os.path.isfile(f"{hyperparameters_path}/{algorithm}.json"):
                auxiliary_data_frame = AutoMLPipelineFeatureGenerator(enable_text_special_features=False,
                                                                      enable_text_ngram_features=False)
                auxiliary_data_frame = auxiliary_data_frame.fit_transform(df)

                train_data = TabularDataset(auxiliary_data_frame)
                predictor = TabularPredictor(label=dataset_info.target_label, eval_metric=scoring, verbosity=0)
                predictor.fit(train_data=train_data, hyperparameters={algorithm: {}})

                training_results = predictor.info()
                Writer.write_json_to_file(
                    path=f"{hyperparameters_path}", results_file_name=f"{algorithm}.json",
                    json_content=training_results["model_info"][algorithm_name]["hyperparameters"])

            with open(f"{hyperparameters_path}/{algorithm}.json", encoding="utf-8") as json_file:
                hyperparameters_model: dict = json.load(json_file)
                hyperparameters[algorithm] = hyperparameters_model

            print(f"Finished hyperparameters tunning: {algorithm}")

        return hyperparameters

    def get_hyperparameters_svm(self, estimator: SVC | SVR, df: pd.DataFrame, dataset_info: DatasetInfo,
                                max_rows=100, fitting_rounds=5) -> dict:
        """Retrieves or performs hyperparameter tuning for an SVM model on the given dataset.

        Parameters
        ----------
        estimator : SVC | SVR
            The SVM estimator to tune the hyperparameters for.
        df : pd.DataFrame
            The dataset to perform hyperparameter tuning on.
        dataset_info : DatasetInfo
            Information about the dataset, including the target label.
        max_rows : int, optional
            The maximum number of rows to use for training when the dataset is larger than this value (default: 100).
        fitting_rounds : int, optional
            The number of fitting rounds to perform when the dataset size exceeds the `max_rows` value (default: 5).

        Returns
        -------
        dict
            A dictionary containing the tuned hyperparameters for the SVM model.
        """
        hyperparameters_path = f"results/hyperparameters/{dataset_info.dataset_name}"
        svm_hyperparameters: dict = {}

        if not os.path.isfile(f"{hyperparameters_path}/SVM.json"):
            grid = GridSearchCV(estimator, self.svm_param_grid, refit=True, cv=5, n_jobs=-1)
            df = discretize_columns_ordinal_encoder(df, [])

            if df.shape[0] > max_rows:
                actual_fitting_rounds, df_sample = fitting_rounds, df.sample(n=max_rows, random_state=42)
            else:
                actual_fitting_rounds, df_sample = 1, df

            for i in range(0, actual_fitting_rounds):
                print(f"{i + 1}/{actual_fitting_rounds}")
                X, y = split_input_target(df_sample, dataset_info.target_label)
                grid.fit(X, y)
                df_sample = df.sample(n=max_rows, random_state=42)

            Writer.write_json_to_file(
                path=f"{hyperparameters_path}", results_file_name="SVM.json", json_content=grid.best_params_)

        with open(f"{hyperparameters_path}/SVM.json", encoding="utf-8") as json_file:
            svm_hyperparameters = json.load(json_file)

        print("Finished hyperparameters tunning: SVM")
        return svm_hyperparameters


if __name__ == "__main__":
    main()
