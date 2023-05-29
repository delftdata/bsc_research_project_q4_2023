import os
# import warnings
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from evaluation.evaluator import Evaluator
from methods.filter import Filter
from methods.wrapper import Wrapper
from processing.imputer import Imputer

# warnings.filterwarnings("ignore")
# os.environ["PYTHONWARNINGS"] = "ignore"


def main():
    algorithm_names: list[tuple[str, str]] = [
        ("GBM", "LightGBM"), ("RF", "RandomForest"), ("LR", "LinearModel"), ("XGB", "XGBoost")]
    runner = Runner(algorithm_names)
    runner.run_bike_sharing()


@dataclass
class DatasetInfo:
    dataset_file: str
    target_label: str
    results_path: str
    eval_metric: Literal["accuracy", "neg_root_mean_squared_error"] = "accuracy"
    file_names: str = ""


class Runner:
    def __init__(self, algorithm_names: list[tuple[str, str]],
                 preprocessing=True, imputation_strategy: Literal["mean", "median"] = "mean",
                 experiment_name="experiment2"):
        self.algorithm_names = algorithm_names
        self.preprocessing = preprocessing
        self.imputation_strategy: Literal["mean", "median"] = imputation_strategy
        self.experiment_name = experiment_name

    def run_arrhythmia(self):
        arrhythmia = DatasetInfo("data/arrhythmia/arrhythmia.csv", "Class",
                                 f"results/{self.experiment_name}/arrhythmia")
        df_arrhythmia = pd.read_csv(arrhythmia.dataset_file, low_memory=False)
        df_arrhythmia = df_arrhythmia.replace("?", np.nan)
        df_arrhythmia = Imputer.impute_mean_or_median(df=df_arrhythmia, strategy=self.imputation_strategy)
        df_arrhythmia = Imputer.impute_most_frequent(df=df_arrhythmia)
        self.evaluate_feature_selection(df_arrhythmia, arrhythmia)

    def run_crop(self):
        crop = DatasetInfo("data/crop", "label", f"results/{self.experiment_name}/crop")
        frames = []
        for i in range(2):
            frames.append(pd.read_csv(f"{crop.dataset_file}/crop{i}.csv", low_memory=False))
        df_crop = pd.concat(frames)
        self.evaluate_feature_selection(df_crop, crop)

    def run_steel_plates_faults(self):
        steel_plates_faults = DatasetInfo("data/steel_plates_faults/steel_plates_faults.csv",
                                          "Class", f"results/{self.experiment_name}/steel_plates_faults")
        df_steel_plates_faults = pd.read_csv(steel_plates_faults.dataset_file, low_memory=False)
        self.evaluate_feature_selection(df_steel_plates_faults, steel_plates_faults)

    def run_bank(self):
        bank = DatasetInfo("data/bank_marketing/bank.csv", "y", f"results/{self.experiment_name}/bank_marketing")
        df_bank = pd.read_csv(bank.dataset_file, low_memory=False)
        df_bank = df_bank.replace("unknown", np.nan)
        df_bank = Imputer.impute_mean_or_median(df=df_bank, strategy=self.imputation_strategy)
        df_bank = Imputer.impute_most_frequent(df=df_bank)
        self.evaluate_feature_selection(df_bank, bank)

    def run_breast_cancer(self):
        breast_cancer = DatasetInfo("data/breast_cancer/breast_cancer.csv", "diagnosis",
                                    f"results/{self.experiment_name}/breast_cancer")
        df_breast_cancer = pd.read_csv(breast_cancer.dataset_file, low_memory=False)
        self.evaluate_feature_selection(df_breast_cancer, breast_cancer)

    def run_census_income(self):
        census_income = DatasetInfo("data/census_income/census_income.csv", "income_label",
                                    f"results/{self.experiment_name}/census_income")
        df_census_income = pd.read_csv(census_income.dataset_file, low_memory=False)
        self.evaluate_feature_selection(df_census_income, census_income)

    def run_character_font_images(self):
        character_font_images = DatasetInfo("data/character_font_images", "font",
                                            f"results/{self.experiment_name}/character_font_images",
                                            file_names="data/character_font_images/font.names")
        df_file_names = pd.read_csv(character_font_images.file_names, low_memory=False, header=None)
        frames = []
        for i, file_name in enumerate(df_file_names[0]):
            print(file_name)
            frames.append(pd.read_csv(f"{character_font_images.dataset_file}/{file_name}"))
            if i == 1:
                break
        df_character_font_images = pd.concat(frames)
        self.evaluate_feature_selection(df_character_font_images, character_font_images)

    def run_internet_ads(self):
        internet_ads = DatasetInfo("data/internet_advertisements/internet_advertisements.csv",
                                   "class", f"results/{self.experiment_name}/internet_advertisements")
        df_internet_ads = pd.read_csv(internet_ads.dataset_file, low_memory=False)
        self.evaluate_feature_selection(df_internet_ads, internet_ads)

    def run_connect_4(self):
        connect_4 = DatasetInfo("data/connect-4/connect-4.csv", "winner", f"results/{self.experiment_name}/connect-4")
        df_connect_4 = pd.read_csv(connect_4.dataset_file, low_memory=False)
        df_connect_4.fillna(np.nan)
        df_connect_4 = Imputer.impute_mean_or_median(df_connect_4, self.imputation_strategy)
        df_connect_4 = Imputer.impute_most_frequent(df_connect_4)
        self.evaluate_feature_selection(df_connect_4, connect_4)

    def run_housing_prices(self):
        housing_prices = DatasetInfo(
            "data/housing_prices/housing_prices.csv", "SalePrice", f"results/{self.experiment_name}/housing_prices",
            eval_metric="neg_root_mean_squared_error")
        df_housing_prices = pd.read_csv(housing_prices.dataset_file, low_memory=False)
        df_housing_prices = df_housing_prices.fillna(np.nan)
        df_housing_prices = Imputer.impute_mean_or_median(df_housing_prices, self.imputation_strategy)
        df_housing_prices = Imputer.impute_most_frequent(df_housing_prices)
        self.evaluate_feature_selection(df_housing_prices, housing_prices)

    def run_nasa_numeric(self):
        nasa_numeric = DatasetInfo(
            "data/nasa_numeric/nasa_numeric.csv", "cat2", f"results/{self.experiment_name}/nasa_numeric",
            eval_metric="neg_root_mean_squared_error")
        df_nasa_numeric = pd.read_csv(nasa_numeric.dataset_file, low_memory=False)
        self.evaluate_feature_selection(df_nasa_numeric, nasa_numeric)

    def run_bike_sharing(self):
        bike_sharing = DatasetInfo("data/bike_sharing/hour.csv", "cnt", f"results/{self.experiment_name}/bike_sharing",
                                   eval_metric="neg_root_mean_squared_error")
        df_bike_sharing = pd.read_csv(bike_sharing.dataset_file, low_memory=False)
        self.evaluate_feature_selection(df_bike_sharing, bike_sharing)

    def evaluate_feature_selection(self, df: pd.DataFrame, dataset: DatasetInfo):
        filter_methods: list[Literal["chi2", "anova"]] = ["chi2", "anova"]
        wrapper_methods: list[Literal["forward_selection", "backward_elimination"]] = [
            "forward_selection", "backward_elimination"]
        selected_features_path = f"{dataset.results_path}/selected_features"

        for method in filter_methods:
            if not os.path.isfile(f"{selected_features_path}/{method}.txt"):
                sorted_features = Filter.rank_features_descending(df, method, dataset.target_label, self.preprocessing)
                write_selected_features(dataset, sorted_features, method)

        for method in wrapper_methods:
            if not os.path.isfile(f"{selected_features_path}/{method}.txt"):
                sorted_features = Wrapper.rank_features_descending(df, method, dataset.target_label,
                                                                   dataset.eval_metric, self.preprocessing)
                write_selected_features(dataset, sorted_features, method)

        methods = filter_methods + wrapper_methods
        evaluator = Evaluator(df, dataset.target_label, "root_mean_squared_error" if dataset.eval_metric ==
                              "neg_root_mean_squared_error" else dataset.eval_metric, self.algorithm_names)

        for method in methods:
            sorted_features = [line.strip() for line in open(f"{selected_features_path}/{method}.txt", "r")]
            performance = evaluator.perform_experiments(sorted_features)
            write_performance(dataset, performance, method)


def write_selected_features(dataset: DatasetInfo, selected_features: list[str], method: str):
    print(f"{method}: {selected_features}")
    for selected_feature in selected_features:
        write_to_file(f"{dataset.results_path}/selected_features", f"{method}.txt", selected_feature)


def write_performance(dataset: DatasetInfo, performance: dict[str, list[float]], method):
    for (algorithm, performance_algorithm) in performance.items():
        print(f"{algorithm}: {performance_algorithm}")
        content = ",".join([str(x) for x in performance_algorithm])
        write_to_file(f"{dataset.results_path}/{method}", f"{algorithm}.txt", content)


def write_to_file(path: str, results_file_name: str, content: str, mode="a+"):
    path_to_file = f"{path}/{results_file_name}"
    try:
        if not os.path.isdir(path):
            os.makedirs(path)
        with open(path_to_file, mode=mode) as file:
            file.write(content + "\n")
        print(f"Updated -{path_to_file}- with content -{content}-")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
