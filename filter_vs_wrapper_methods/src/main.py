import os
import sys
import warnings
from dataclasses import dataclass
from multiprocessing import Pool
from typing import Literal

import pandas as pd
from evaluation.evaluator import Evaluator
from methods.filter import rank_features_descending_filter
from methods.wrapper import rank_features_descending_wrapper
from numpy import nan
from processing.imputer import (drop_missing_values, impute_mean_or_median,
                                impute_most_frequent)
from processing.preprocessing import convert_to_actual_type
from processing.splitter import (
    drop_features, drop_features_with_negative_values,
    split_categorical_discrete_continuous_features)

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"


def main():
    algorithm_names: list[tuple[str, str]] = [
        ("GBM", "LightGBM"), ("RF", "RandomForest"), ("LR", "LinearModel"), ("XGB", "XGBoost")]
    experiment_name = sys.argv[1]
    print(f"Experiment_name: {experiment_name}")
    preprocessing = experiment_name != "experiment1" and experiment_name != "experiment3"
    imputation_strategy: Literal["mean", "median"] = "median"
    normalization = True
    runner = Runner(algorithm_names, experiment_name, preprocessing, imputation_strategy, normalization)
    dataset = sys.argv[2]
    print(f"Dataset: {dataset}")
    if dataset == "small":
        runner.run_experiment_on_small_datasets_in_parallel()
    elif dataset == "big":
        runner.run_experiment_on_big_datasets_sequentially()
    else:
        runner.run_experiment_on_dataset(dataset)


@dataclass
class DatasetInfo:
    dataset_file: str
    target_label: str
    results_path: str
    eval_metric: Literal["accuracy", "neg_root_mean_squared_error"] = "accuracy"
    file_names: str = ""


class Runner:
    def __init__(self, algorithm_names: list[tuple[str, str]], experiment_name: str,
                 preprocessing=True, imputation_strategy: Literal["mean", "median"] = "mean", normalization=True):
        self.algorithm_names = algorithm_names
        self.experiment_name = experiment_name
        self.preprocessing = preprocessing
        self.imputation_strategy: Literal["mean", "median"] = imputation_strategy
        self.normalization = normalization
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
        if missing_values:
            if self.experiment_name == "experiment2" or self.experiment_name == "experiment4":
                df = impute_mean_or_median(df, strategy=self.imputation_strategy)
                df = impute_most_frequent(df)
            elif self.experiment_name == "experiment3":
                df = drop_missing_values(df)
        df = convert_to_actual_type(df)
        return df

    def run_experiment_on_dataset(self, dataset: str):
        self.runner_dictionary[dataset]()

    def run_experiment_on_small_datasets_in_parallel(self):
        with Pool() as pool:
            small_datasets = ["bank_marketing", "breast_cancer", "steel_plates_faults",
                              "housing_prices", "bike_sharing", "census_income", "connect_4", "nasa_numeric"]
            pool.map(self.run_experiment_on_dataset, small_datasets)

    def run_experiment_on_big_datasets_sequentially(self):
        big_datasets = ["arrhythmia", "crop", "character_font_images", "internet_ads"]
        for big_dataset in big_datasets:
            self.run_experiment_on_dataset(dataset=big_dataset)

    def run_experiment3(self, df: pd.DataFrame, dataset_info: DatasetInfo, min_columns=2):
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
            self.evaluate_feature_selection(df_chi2, dataset_info, methods=["chi2"])
        if df_anova.columns.size > min_columns:
            self.evaluate_feature_selection(df_anova, dataset_info, methods=["anova"])
        if df_forward_selection.columns.size > min_columns:
            self.evaluate_feature_selection(df_forward_selection, dataset_info, methods=["forward_selection"])
        if df_backward_elimination.columns.size > min_columns:
            self.evaluate_feature_selection(df_backward_elimination, dataset_info, methods=["backward_elimination"])

    def run_experiment4(self, df: pd.DataFrame, dataset_info: DatasetInfo, min_columns=2):
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

        # print(df_categorical.dtypes)
        # print(df_categorical)
        # print(df_discrete.dtypes)
        # print(df_discrete)
        # print(df_continuous.dtypes)
        # print(df_continuous)

        if df_categorical.columns.size > min_columns:
            print(f"Evaluating categorical features: {df_categorical.shape}.")
            self.evaluate_feature_selection(df_categorical, dataset_info_categorical)
            print(f"Finished evaluating categorical features.")
        if df_discrete.columns.size > min_columns:
            print(f"Evaluating discrete features: {df_discrete.shape}.")
            self.evaluate_feature_selection(df_discrete, dataset_info_discrete)
            print(f"Finished evaluating discrete features.")
        if df_continuous.columns.size > min_columns:
            print(f"Evaluating continuous features: {df_continuous.shape}.")
            self.evaluate_feature_selection(df_continuous, dataset_info_continuous)
            print(f"Finished evaluating continuous features.")

    def run_bank_marketing(self):
        bank = DatasetInfo("data/bank_marketing/bank.csv", "y",
                           f"results/{self.experiment_name}/bank_marketing")
        df_bank = pd.read_csv(bank.dataset_file, low_memory=False)
        df_bank = df_bank.replace("unknown", nan)
        df_bank = self.prepare_data_frame(df=df_bank, missing_values=True)
        # print(df_bank.dtypes)
        # print(df_bank)
        if self.experiment_name == "experiment3":
            self.run_experiment3(df=df_bank, dataset_info=bank)
        elif self.experiment_name == "experiment4":
            self.run_experiment4(df=df_bank, dataset_info=bank)
        else:
            self.evaluate_feature_selection(df=df_bank, dataset_info=bank)

    def run_breast_cancer(self):
        breast_cancer = DatasetInfo("data/breast_cancer/breast_cancer.csv", "diagnosis",
                                    f"results/{self.experiment_name}/breast_cancer")
        df_breast_cancer = pd.read_csv(breast_cancer.dataset_file, low_memory=False)
        df_breast_cancer = self.prepare_data_frame(df=df_breast_cancer)
        # print(df_breast_cancer.dtypes)
        # print(df_breast_cancer)
        if self.experiment_name == "experiment3":
            self.run_experiment3(df=df_breast_cancer, dataset_info=breast_cancer)
        elif self.experiment_name == "experiment4":
            self.run_experiment4(df=df_breast_cancer, dataset_info=breast_cancer)
        else:
            self.evaluate_feature_selection(df=df_breast_cancer, dataset_info=breast_cancer)

    def run_steel_plates_faults(self):
        steel_plates_faults = DatasetInfo("data/steel_plates_faults/steel_plates_faults.csv",
                                          "Class", f"results/{self.experiment_name}/steel_plates_faults")
        df_steel_plates_faults = pd.read_csv(steel_plates_faults.dataset_file, low_memory=False)
        df_steel_plates_faults = self.prepare_data_frame(df=df_steel_plates_faults)
        # print(df_steel_plates_faults.dtypes)
        # print(df_steel_plates_faults)
        if self.experiment_name == "experiment3":
            self.run_experiment3(df=df_steel_plates_faults, dataset_info=steel_plates_faults)
        elif self.experiment_name == "experiment4":
            self.run_experiment4(df=df_steel_plates_faults, dataset_info=steel_plates_faults)
        else:
            self.evaluate_feature_selection(df=df_steel_plates_faults, dataset_info=steel_plates_faults)

    def run_housing_prices(self):
        housing_prices = DatasetInfo(
            "data/housing_prices/housing_prices.csv", "SalePrice", f"results/{self.experiment_name}/housing_prices",
            eval_metric="neg_root_mean_squared_error")
        df_housing_prices = pd.read_csv(housing_prices.dataset_file, low_memory=False)
        df_housing_prices = df_housing_prices.fillna(nan)
        df_housing_prices = self.prepare_data_frame(df=df_housing_prices, missing_values=True)
        # print(df_housing_prices.dtypes)
        # print(df_housing_prices)
        if self.experiment_name == "experiment3":
            self.run_experiment3(df=df_housing_prices, dataset_info=housing_prices)
        elif self.experiment_name == "experiment4":
            self.run_experiment4(df=df_housing_prices, dataset_info=housing_prices)
        else:
            self.evaluate_feature_selection(df=df_housing_prices, dataset_info=housing_prices)

    def run_bike_sharing(self):
        bike_sharing = DatasetInfo("data/bike_sharing/hour.csv", "cnt", f"results/{self.experiment_name}/bike_sharing",
                                   eval_metric="neg_root_mean_squared_error")
        df_bike_sharing = pd.read_csv(bike_sharing.dataset_file, low_memory=False)
        df_bike_sharing = self.prepare_data_frame(df=df_bike_sharing)
        # print(df_bike_sharing.dtypes)
        # print(df_bike_sharing)
        if self.experiment_name == "experiment3":
            self.run_experiment3(df=df_bike_sharing, dataset_info=bike_sharing)
        elif self.experiment_name == "experiment4":
            self.run_experiment4(df=df_bike_sharing, dataset_info=bike_sharing)
        else:
            self.evaluate_feature_selection(df=df_bike_sharing, dataset_info=bike_sharing)

    def run_census_income(self):
        census_income = DatasetInfo("data/census_income/census_income.csv", "income_label",
                                    f"results/{self.experiment_name}/census_income")
        df_census_income = pd.read_csv(census_income.dataset_file, low_memory=False)
        df_census_income = df_census_income.fillna(nan)
        df_census_income = self.prepare_data_frame(df=df_census_income, missing_values=True)
        # print(df_census_income.dtypes)
        # print(df_census_income)
        if self.experiment_name == "experiment3":
            self.run_experiment3(df=df_census_income, dataset_info=census_income)
        elif self.experiment_name == "experiment4":
            self.run_experiment4(df=df_census_income, dataset_info=census_income)
        else:
            self.evaluate_feature_selection(df=df_census_income, dataset_info=census_income)

    def run_connect_4(self):
        def map_game(value: float) -> Literal["win", "loss", "tie"]:
            if value == -1.0:
                return "loss"
            if value == 0.0:
                return "tie"
            return "win"

        connect_4 = DatasetInfo("data/connect-4/connect-4.csv", "winner", f"results/{self.experiment_name}/connect-4")
        df_connect_4 = pd.read_csv(connect_4.dataset_file, low_memory=False)
        df_connect_4 = df_connect_4.fillna(nan)
        df_connect_4 = self.prepare_data_frame(df=df_connect_4, missing_values=True)
        df_connect_4[connect_4.target_label] = df_connect_4[connect_4.target_label].apply(lambda x: map_game(x))
        df_connect_4[connect_4.target_label] = df_connect_4[connect_4.target_label].astype("category")
        # print(df_connect_4.dtypes)
        # print(df_connect_4)
        if self.experiment_name == "experiment3":
            self.run_experiment3(df=df_connect_4, dataset_info=connect_4)
        elif self.experiment_name == "experiment4":
            self.run_experiment4(df=df_connect_4, dataset_info=connect_4)
        else:
            self.evaluate_feature_selection(df=df_connect_4, dataset_info=connect_4)

    def run_nasa_numeric(self):
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
        character_font_images = DatasetInfo("data/character_font_images", "font",
                                            f"results/{self.experiment_name}/character_font_images",
                                            file_names="data/character_font_images/font.names")
        df_file_names = pd.read_csv(character_font_images.file_names, low_memory=False, header=None)
        frames = []
        for i, file_name in enumerate(df_file_names[0]):
            # print(file_name)
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
        methods: list[Literal["chi2", "anova", "forward_selection", "backward_elimination"]] =
            ["chi2", "anova", "forward_selection", "backward_elimination"]):

        selected_features_path = f"{dataset_info.results_path}/selected_features"

        for method in methods:
            if not os.path.isfile(f"{selected_features_path}/{method}.txt"):

                if method == "chi2" or method == "anova":
                    sorted_features, runtime = rank_features_descending_filter(
                        df, method, dataset_info.target_label, self.preprocessing)
                else:
                    sorted_features, runtime = rank_features_descending_wrapper(
                        df, method, dataset_info.target_label, dataset_info.eval_metric, self.preprocessing)

                write_runtime(dataset_info, runtime, method)
                write_selected_features(dataset_info, sorted_features, method)
            print(f"Finished feature selection, {method}.")

        evaluator = Evaluator(df, dataset_info.target_label, "root_mean_squared_error" if dataset_info.eval_metric ==
                              "neg_root_mean_squared_error" else dataset_info.eval_metric, self.algorithm_names)

        for method in methods:
            sorted_features = [line.strip() for line in open(f"{selected_features_path}/{method}.txt", "r")]
            performance = evaluator.perform_experiments(sorted_features)
            print(f"Autogluon finished evaluating the features selected by: {method}")
            write_performance(dataset_info, performance, method)


def write_selected_features(dataset_info: DatasetInfo, selected_features: list[str], method: str):
    # print(f"{method}: {selected_features}")
    for selected_feature in selected_features:
        write_to_file(f"{dataset_info.results_path}/selected_features", f"{method}.txt", selected_feature)


def write_runtime(dataset_info: DatasetInfo, runtime: float, method: str):
    # print(f"Runtime: {method} - {runtime}")
    write_to_file(f"{dataset_info.results_path}/runtime", f"{method}.txt", str(runtime))


def write_performance(dataset_info: DatasetInfo, performance: dict[str, list[float]], method):
    for (algorithm, performance_algorithm) in performance.items():
        # print(f"{algorithm}: {performance_algorithm}")
        content = ",".join([str(x) for x in performance_algorithm])
        write_to_file(f"{dataset_info.results_path}/{method}", f"{algorithm}.txt", content)


def write_to_file(path: str, results_file_name: str, content: str, mode="a+"):
    path_to_file = f"{path}/{results_file_name}"
    try:
        if not os.path.isdir(path):
            os.makedirs(path)
        with open(path_to_file, mode=mode) as file:
            file.write(content + "\n")
        # print(f"Updated -{path_to_file}- with content -{content}-")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
