from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass
from multiprocessing import Pool
from typing import Literal

import pandas as pd
from numpy import nan

from evaluation.evaluator import Evaluator
from methods.filter import rank_features_descending_filter
from methods.wrapper import rank_features_descending_wrapper
from processing.imputer import (drop_missing_values, impute_mean_or_median,
                                impute_most_frequent)
from processing.preprocessing import convert_to_actual_type
from processing.splitter import (
    drop_features, drop_features_with_negative_values,
    split_categorical_discrete_continuous_features)

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


@dataclass
class DatasetInfo:
    """
    A dataclass used to represent the dataset information needed to perform the experiments.

    Attributes
    ----------
    dataset_file : str
        Name of the dataset (usually what comes before .csv).
    target_label : str
        Name of the target label.
    results_path : str
        Path to the file where the results obtained from performing the experiments on the dataset should be stored.
    eval_metric : Literal["accuracy", "neg_root_mean_squared_error"], optional
        Evaluation metric of the dataset (default: "accuracy").
        For classification tasks, use "accuracy".
        For regression tasks, use "neg_root_mean_squared_error".
    file_names : str, optional
        Only used by the character font images dataset, since it is split into multiple files (default: "").
    """
    dataset_file: str
    target_label: str
    results_path: str
    eval_metric: Literal["accuracy", "neg_root_mean_squared_error"] = "accuracy"
    file_names: str = ""


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
    runner_dictionary : dict[str, Callable]
        Dictionary that binds the dataset name to a method used to run the experiments on the specific dataset.
    """

    def __init__(
            self, algorithm_names: list[tuple[str, str]],
            svm_param_grid: list[dict[str, list[str | int | float]]],
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
        self.runner_dictionary = {
            "bank_marketing": self.run_bank_marketing,
            "breast_cancer": self.run_breast_cancer,
            "steel_plates_faults": self.run_steel_plates_faults,
            "housing_prices": self.run_housing_prices,
            "bike_sharing": self.run_bike_sharing,
            "census_income": self.run_census_income,
            "connect_4": self.run_connect_4,
            "arrhythmia": self.run_arrhythmia,
            "crop": self.run_crop,
            "character_font_images": self.run_character_font_images,
            "internet_ads": self.run_internet_ads,
            "nasa_numeric": self.run_nasa_numeric,
        }

    def prepare_data_frame(self, df: pd.DataFrame, missing_values=False):
        """Prepares the given DataFrame for further processing in the experiments.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to be prepared.
        missing_values : bool, optional
            Flag indicating whether to handle missing values in the DataFrame (default: False).
            If True, missing values will be handled based on the configured imputation strategy.

        Returns
        -------
        pd.DataFrame
            The prepared DataFrame.
        """
        if missing_values:
            if self.experiment_name in set(["experiment2", "experiment4"]):
                df = impute_mean_or_median(df, strategy=self.imputation_strategy)
                df = impute_most_frequent(df)
            elif self.experiment_name == "experiment3":
                df = drop_missing_values(df)
        df = convert_to_actual_type(df)
        return df

    def run_experiment_on_dataset(self, dataset: str):
        """Runs the experiment on a specific dataset.

        Parameters
        ----------
        dataset : str
            The name of the dataset to run the experiment on.

        Raises
        ------
        KeyError
            If the specified dataset does not exist in the runner dictionary.
        """
        self.runner_dictionary[dataset]()

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
            print(f"Evaluating categorical features: {df_categorical.shape}.")
            self.evaluate_feature_selection(df_categorical, dataset_info_categorical)
            print("Finished evaluating categorical features.")
        if df_discrete.columns.size > min_columns:
            print(f"Evaluating discrete features: {df_discrete.shape}.")
            self.evaluate_feature_selection(df_discrete, dataset_info_discrete)
            print("Finished evaluating discrete features.")
        if df_continuous.columns.size > min_columns:
            print(f"Evaluating continuous features: {df_continuous.shape}.")
            self.evaluate_feature_selection(df_continuous, dataset_info_continuous)
            print("Finished evaluating continuous features.")

    def run_bank_marketing(self):
        """Runs the experiment on the `bank_marketing` dataset.

        Raises
        ------
        FileNotFoundError
            If the dataset file for `bank_marketing` cannot be found.
        """
        bank = DatasetInfo("data/bank_marketing/bank.csv", "y",
                           f"results/{self.experiment_name}/bank_marketing")
        df_bank = pd.read_csv(bank.dataset_file, low_memory=False)
        df_bank = df_bank.replace("unknown", nan)
        df_bank = self.prepare_data_frame(df=df_bank, missing_values=True)
        if self.experiment_name == "experiment3":
            self.run_experiment3(df=df_bank, dataset_info=bank)
        elif self.experiment_name == "experiment4":
            self.run_experiment4(df=df_bank, dataset_info=bank)
        else:
            self.evaluate_feature_selection(df=df_bank, dataset_info=bank)

    def run_breast_cancer(self):
        """Runs the experiment on the `breast_cancer` dataset.

        Raises
        ------
        FileNotFoundError
            If the dataset file for `breast_cancer` cannot be found.
        """
        breast_cancer = DatasetInfo("data/breast_cancer/breast_cancer.csv", "diagnosis",
                                    f"results/{self.experiment_name}/breast_cancer")
        df_breast_cancer = pd.read_csv(breast_cancer.dataset_file, low_memory=False)
        df_breast_cancer = self.prepare_data_frame(df=df_breast_cancer)
        if self.experiment_name == "experiment3":
            self.run_experiment3(df=df_breast_cancer, dataset_info=breast_cancer)
        elif self.experiment_name == "experiment4":
            self.run_experiment4(df=df_breast_cancer, dataset_info=breast_cancer)
        else:
            self.evaluate_feature_selection(df=df_breast_cancer, dataset_info=breast_cancer)

    def run_steel_plates_faults(self):
        """Runs the experiment on the `steel_plates_faults` dataset.

        Raises
        ------
        FileNotFoundError
            If the dataset file for `steel_plates_faults` cannot be found.
        """
        steel_plates_faults = DatasetInfo("data/steel_plates_faults/steel_plates_faults.csv",
                                          "Class", f"results/{self.experiment_name}/steel_plates_faults")
        df_steel_plates_faults = pd.read_csv(steel_plates_faults.dataset_file, low_memory=False)
        df_steel_plates_faults = self.prepare_data_frame(df=df_steel_plates_faults)
        if self.experiment_name == "experiment3":
            self.run_experiment3(df=df_steel_plates_faults, dataset_info=steel_plates_faults)
        elif self.experiment_name == "experiment4":
            self.run_experiment4(df=df_steel_plates_faults, dataset_info=steel_plates_faults)
        else:
            self.evaluate_feature_selection(df=df_steel_plates_faults, dataset_info=steel_plates_faults)

    def run_housing_prices(self):
        """Runs the experiment on the `housing_prices` dataset.

        Raises
        ------
        FileNotFoundError
            If the dataset file for `housing_prices` cannot be found.
        """
        housing_prices = DatasetInfo(
            "data/housing_prices/housing_prices.csv", "SalePrice", f"results/{self.experiment_name}/housing_prices",
            eval_metric="neg_root_mean_squared_error")
        df_housing_prices = pd.read_csv(housing_prices.dataset_file, low_memory=False)
        df_housing_prices = df_housing_prices.fillna(nan)
        df_housing_prices = self.prepare_data_frame(df=df_housing_prices, missing_values=True)
        if self.experiment_name == "experiment3":
            self.run_experiment3(df=df_housing_prices, dataset_info=housing_prices)
        elif self.experiment_name == "experiment4":
            self.run_experiment4(df=df_housing_prices, dataset_info=housing_prices)
        else:
            self.evaluate_feature_selection(df=df_housing_prices, dataset_info=housing_prices)

    def run_bike_sharing(self):
        """Runs the experiment on the `bike_sharing` dataset.

        Raises
        ------
        FileNotFoundError
            If the dataset file for `bike_sharing` cannot be found.
        """
        bike_sharing = DatasetInfo("data/bike_sharing/hour.csv", "cnt", f"results/{self.experiment_name}/bike_sharing",
                                   eval_metric="neg_root_mean_squared_error")
        df_bike_sharing = pd.read_csv(bike_sharing.dataset_file, low_memory=False)
        df_bike_sharing = self.prepare_data_frame(df=df_bike_sharing)
        if self.experiment_name == "experiment3":
            self.run_experiment3(df=df_bike_sharing, dataset_info=bike_sharing)
        elif self.experiment_name == "experiment4":
            self.run_experiment4(df=df_bike_sharing, dataset_info=bike_sharing)
        else:
            self.evaluate_feature_selection(df=df_bike_sharing, dataset_info=bike_sharing)

    def run_census_income(self):
        """Runs the experiment on the `census_income` dataset.

        Raises
        ------
        FileNotFoundError
            If the dataset file for `census_income` cannot be found.
        """
        census_income = DatasetInfo("data/census_income/census_income.csv", "income_label",
                                    f"results/{self.experiment_name}/census_income")
        df_census_income = pd.read_csv(census_income.dataset_file, low_memory=False)
        df_census_income = df_census_income.fillna(nan)
        df_census_income = self.prepare_data_frame(df=df_census_income, missing_values=True)
        if self.experiment_name == "experiment3":
            self.run_experiment3(df=df_census_income, dataset_info=census_income)
        elif self.experiment_name == "experiment4":
            self.run_experiment4(df=df_census_income, dataset_info=census_income)
        else:
            self.evaluate_feature_selection(df=df_census_income, dataset_info=census_income)

    def run_connect_4(self):
        """Runs the experiment on the `connect-4` dataset.

        Raises
        ------
        FileNotFoundError
            If the dataset file for `connect-4` cannot be found.
        """

        def map_game(value: float) -> Literal["win", "loss", "tie"]:
            """Maps a numerical game outcome value to its corresponding label.

            Parameters
            ----------
            value : float
                The numerical value representing the game outcome.

            Returns
            -------
            Literal["win", "loss", "tie"]
                The mapped label corresponding to the game outcome:
                - If `value` is -1.0, returns "loss".
                - If `value` is 0.0, returns "tie".
                - For any other value, returns "win".
            """
            if value == -1.0:
                return "loss"
            if value == 0.0:
                return "tie"
            return "win"

        connect_4 = DatasetInfo("data/connect-4/connect-4.csv", "winner", f"results/{self.experiment_name}/connect-4")
        df_connect_4 = pd.read_csv(connect_4.dataset_file, low_memory=False)
        df_connect_4 = df_connect_4.fillna(nan)
        df_connect_4 = self.prepare_data_frame(df=df_connect_4, missing_values=True)
        df_connect_4[connect_4.target_label] = df_connect_4[connect_4.target_label].apply(map_game)
        df_connect_4[connect_4.target_label] = df_connect_4[connect_4.target_label].astype("category")
        if self.experiment_name == "experiment3":
            self.run_experiment3(df=df_connect_4, dataset_info=connect_4)
        elif self.experiment_name == "experiment4":
            self.run_experiment4(df=df_connect_4, dataset_info=connect_4)
        else:
            self.evaluate_feature_selection(df=df_connect_4, dataset_info=connect_4)

    def run_nasa_numeric(self):
        """Runs the experiment on the `nasa_numeric` dataset.

        Raises
        ------
        FileNotFoundError
            If the dataset file for `nasa_numeric` cannot be found.
        """
        nasa_numeric = DatasetInfo(
            "data/nasa_numeric/nasa_numeric.csv", "act_effort", f"results/{self.experiment_name}/nasa_numeric",
            eval_metric="neg_root_mean_squared_error")
        df_nasa_numeric = pd.read_csv(nasa_numeric.dataset_file, low_memory=False)
        df_nasa_numeric = df_nasa_numeric.fillna(nan)
        df_nasa_numeric = self.prepare_data_frame(df=df_nasa_numeric, missing_values=True)
        if self.experiment_name == "experiment3":
            self.run_experiment3(df=df_nasa_numeric, dataset_info=nasa_numeric)
        elif self.experiment_name == "experiment4":
            self.run_experiment4(df=df_nasa_numeric, dataset_info=nasa_numeric)
        else:
            self.evaluate_feature_selection(df=df_nasa_numeric, dataset_info=nasa_numeric)

    def run_arrhythmia(self):
        """Runs the experiment on the `arrhythmia` dataset.

        Raises
        ------
        FileNotFoundError
            If the dataset file for `arrhythmia` cannot be found.
        """
        arrhythmia = DatasetInfo("data/arrhythmia/arrhythmia.csv", "Class",
                                 f"results/{self.experiment_name}/arrhythmia")
        df_arrhythmia = pd.read_csv(arrhythmia.dataset_file, low_memory=False)
        df_arrhythmia = df_arrhythmia.replace("?", nan)
        df_arrhythmia = self.prepare_data_frame(df=df_arrhythmia, missing_values=True)
        if self.experiment_name == "experiment3":
            self.run_experiment3(df=df_arrhythmia, dataset_info=arrhythmia)
        elif self.experiment_name == "experiment4":
            self.run_experiment4(df=df_arrhythmia, dataset_info=arrhythmia)
        else:
            self.evaluate_feature_selection(df=df_arrhythmia, dataset_info=arrhythmia)

    def run_crop(self):
        """Runs the experiment on the `crop` dataset.

        Raises
        ------
        FileNotFoundError
            If the dataset files for `crop` cannot be found.
        """
        crop = DatasetInfo("data/crop", "label", f"results/{self.experiment_name}/crop")
        frames = []
        for i in range(10):
            frames.append(pd.read_csv(f"{crop.dataset_file}/crop{i}.csv", low_memory=False))
        df_crop = pd.concat(frames)
        df_crop = self.prepare_data_frame(df=df_crop)
        if self.experiment_name == "experiment3":
            self.run_experiment3(df=df_crop, dataset_info=crop)
        elif self.experiment_name == "experiment4":
            self.run_experiment4(df=df_crop, dataset_info=crop)
        else:
            self.evaluate_feature_selection(df=df_crop, dataset_info=crop)

    def run_character_font_images(self):
        """Runs the experiment on the `character_font_images` dataset.

        Raises
        ------
        FileNotFoundError
            If the dataset files for `character_font_images` cannot be found.
        """
        character_font_images = DatasetInfo("data/character_font_images", "font",
                                            f"results/{self.experiment_name}/character_font_images",
                                            file_names="data/character_font_images/font.names")
        df_file_names = pd.read_csv(character_font_images.file_names, low_memory=False, header=None)
        frames = []
        for file_name in df_file_names[0]:
            frames.append(pd.read_csv(f"{character_font_images.dataset_file}/{file_name}"))
        df_character_font_images = pd.concat(frames)
        df_character_font_images = self.prepare_data_frame(df=df_character_font_images)
        if self.experiment_name == "experiment3":
            self.run_experiment3(df=df_character_font_images, dataset_info=character_font_images)
        elif self.experiment_name == "experiment4":
            self.run_experiment4(df=df_character_font_images, dataset_info=character_font_images)
        else:
            self.evaluate_feature_selection(df=df_character_font_images, dataset_info=character_font_images)

    def run_internet_ads(self):
        """Runs the experiment on the `internet_advertisements` dataset.

        Raises
        ------
        FileNotFoundError
            If the dataset file for `internet_advertisements` cannot be found.
        """
        internet_ads = DatasetInfo("data/internet_advertisements/internet_advertisements.csv",
                                   "class", f"results/{self.experiment_name}/internet_advertisements")
        df_internet_ads = pd.read_csv(internet_ads.dataset_file, low_memory=False)
        df_internet_ads = self.prepare_data_frame(df=df_internet_ads)
        if self.experiment_name == "experiment3":
            self.run_experiment3(df=df_internet_ads, dataset_info=internet_ads)
        elif self.experiment_name == "experiment4":
            self.run_experiment4(df=df_internet_ads, dataset_info=internet_ads)
        else:
            self.evaluate_feature_selection(df=df_internet_ads, dataset_info=internet_ads)

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

                if method in ("chi2", "anova"):
                    sorted_features, runtime = rank_features_descending_filter(
                        df, method, dataset_info.target_label, self.preprocessing, self.normalization)
                else:
                    sorted_features, runtime = rank_features_descending_wrapper(
                        df, method, dataset_info.target_label, dataset_info.eval_metric, self.preprocessing, self.normalization)

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
