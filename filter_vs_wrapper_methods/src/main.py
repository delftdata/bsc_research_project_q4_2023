import os
import warnings
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from evaluation.evaluator import Evaluator
from methods.filter import Filter
from methods.wrapper import Wrapper
from plotting.plotter import Plotter
from preprocessing.imputer import Imputer

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"


@dataclass
class DatasetInfo:
    dataset_file: str
    target_label: str
    results_path: str
    eval_metric: Literal["accuracy", "neg_mean_squared_error"] = "accuracy"
    scoring: Literal["Accuracy", "Mean Squared Error"] = "Accuracy"
    file_names: str = ""


def main():
    algorithm_names = [("GBM", "LightGBM"), ("RF", "RandomForest"), ("LR", "LinearModel"), ("XGB", "XGBoost")]
    # run_arrhythmia(algorithm_names)
    run_steel_plates_faults(algorithm_names)
    # run_crop(algorithm_names)


def run_arrhythmia(algorithm_names, preprocessing=True, imputation_strategy: Literal["mean", "median"] = "mean"):
    arrhythmia = DatasetInfo("data/arrhythmia/arrhythmia.csv", "Class", "results/arrhythmia")
    df_arrhythmia = pd.read_csv(arrhythmia.dataset_file, low_memory=False)
    df_arrhythmia = df_arrhythmia.replace("?", np.nan)
    df_arrhythmia = Imputer.impute_mean_or_median(df=df_arrhythmia, strategy=imputation_strategy)
    df_arrhythmia = Imputer.impute_most_frequent(df=df_arrhythmia)
    evaluate_feature_selection(df_arrhythmia, arrhythmia, algorithm_names, preprocessing)


def run_crop(algorithm_names, preprocessing=True):
    crop = DatasetInfo("data/crop", "label", "results/crop")
    frames = []
    for i in range(2):
        frames.append(pd.read_csv(f"{crop.dataset_file}/crop{i}.csv", low_memory=False))
    df_crop = pd.concat(frames)
    evaluate_feature_selection(df_crop, crop, algorithm_names, preprocessing)


def run_steel_plates_faults(algorithm_names, preprocessing=True):
    steel_plates_faults = DatasetInfo("data/steel_plates_faults/steel_plates_faults.csv",
                                      "Class", "results/steel_plates_faults")
    df_steel_plates_faults = pd.read_csv(steel_plates_faults.dataset_file, low_memory=False)
    evaluate_feature_selection(df_steel_plates_faults, steel_plates_faults, algorithm_names, preprocessing)


def run_bank(algorithm_names, preprocessing=True, imputation_strategy: Literal["mean", "median"] = "mean"):
    bank = DatasetInfo("data/bank_marketing/bank.csv", "y", "results/bank_marketing")
    df_bank = pd.read_csv(bank.dataset_file, low_memory=False)
    df_bank = df_bank.replace("unknown", np.nan)
    df_bank = Imputer.impute_mean_or_median(df=df_bank, strategy=imputation_strategy)
    df_bank = Imputer.impute_most_frequent(df=df_bank)
    evaluate_feature_selection(df_bank, bank, algorithm_names, preprocessing)


def run_breast_cancer(algorithm_names, preprocessing=True):
    breast_cancer = DatasetInfo("data/breast_cancer/breast_cancer.csv", "diagnosis", "results/breast_cancer")
    df_breast_cancer = pd.read_csv(breast_cancer.dataset_file, low_memory=False)
    evaluate_feature_selection(df_breast_cancer, breast_cancer, algorithm_names, preprocessing)


def run_census_income(algorithm_names, preprocessing=True):
    census_income = DatasetInfo("data/census_income/census_income.csv", "income_label", "results/census_income")
    df_census_income = pd.read_csv(census_income.dataset_file, low_memory=False)
    evaluate_feature_selection(df_census_income, census_income, algorithm_names, preprocessing)


def run_character_font_images(algorithm_names, preprocessing=True):
    character_font_images = DatasetInfo("data/character_font_images", "font", "results/character_font_images",
                                        file_names="data/character_font_images/font.names")
    df_file_names = pd.read_csv(character_font_images.file_names, low_memory=False, header=None)
    frames = []
    for i, file_name in enumerate(df_file_names[0]):
        print(file_name)
        frames.append(pd.read_csv(f"{character_font_images.dataset_file}/{file_name}"))
        if i == 1:
            break
    df_character_font_images = pd.concat(frames)
    evaluate_feature_selection(df_character_font_images, character_font_images, algorithm_names, preprocessing)


def run_internet_ads(algorithm_names, preprocessing=True):
    internet_ads = DatasetInfo("data/internet_advertisements/internet_advertisements.csv",
                               "class", "results/internet_advertisements")
    df_internet_ads = pd.read_csv(internet_ads.dataset_file, low_memory=False)
    evaluate_feature_selection(df_internet_ads, internet_ads, algorithm_names, preprocessing)


def run_connect_4(algorithm_names, preprocessing=True):
    connect_4 = DatasetInfo("data/connect-4/connect-4.csv", "winner", "results/connect-4")
    df_connect_4 = pd.read_csv(connect_4.dataset_file, low_memory=False)
    evaluate_feature_selection(df_connect_4, connect_4, algorithm_names, preprocessing)


def run_housing_prices(algorithm_names, preprocessing=True):
    housing_prices = DatasetInfo("data/housing_prices/housing_prices.csv", "SalePrice", "results/housing_prices",
                                 eval_metric="neg_mean_squared_error", scoring="Mean Squared Error")
    df_housing_prices = pd.read_csv(housing_prices.dataset_file, low_memory=False)
    evaluate_feature_selection(df_housing_prices, housing_prices, algorithm_names, preprocessing)


def run_nasa_numeric(algorithm_names, preprocessing=True):
    nasa_numeric = DatasetInfo("data/nasa_numeric/nasa_numeric.csv", "cat2", "results/nasa_numeric",
                               eval_metric="neg_mean_squared_error", scoring="Mean Squared Error")
    df_nasa_numeric = pd.read_csv(nasa_numeric.dataset_file, low_memory=False)
    evaluate_feature_selection(df_nasa_numeric, nasa_numeric, algorithm_names, preprocessing)


def evaluate_feature_selection(
        df: pd.DataFrame, dataset: DatasetInfo, algorithm_names: list[tuple[str, str]], preprocessing=True):

    chi2 = Filter(df, "Chi2", dataset.target_label, preprocessing)
    anova = Filter(df, "ANOVA", dataset.target_label, preprocessing)
    # forward_selection = Wrapper(df, "Forward Selection", dataset.target_label, dataset.eval_metric)
    # backward_elimination = Wrapper(df, "Backward Elimination", dataset.target_label, dataset.eval_metric)

    print(chi2.perform_feature_selection(df, 1.0).head())
    print(anova.perform_feature_selection(df, 1.0).head())

    evaluator = Evaluator(df, dataset.target_label, dataset.eval_metric, algorithm_names)

    performances_chi2: dict[str, list[tuple[float, float]]] = evaluator.perform_experiments(method=chi2)  # type: ignore
    write_performance(dataset, performances_chi2, "chi2")
    performances_anova = evaluator.perform_experiments(method=anova)  # type: ignore
    write_performance(dataset, performances_anova, "anova")


def write_performance(dataset: DatasetInfo, performances: dict[str, list[tuple[float, float]]], method_name):
    for (algorithm, performance_algorithm) in performances.items():
        print(f"{algorithm}: {performance_algorithm}")
        # plot_results(dataset.results_path, performance_algorithm, dataset.scoring, algorithm)
        write_to_file(f"{dataset.results_path}/{method_name}", f"{algorithm}.txt", str(performance_algorithm))


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


def plot_results(path: str, performance_algorithm: dict[str, list[float]], scoring, algorithm: str):
    try:
        if not os.path.isdir(path):
            os.makedirs(path)
        algorithm_plot = Plotter.plot_metric_sea_born(performance=performance_algorithm, scoring=scoring)
        figure = algorithm_plot.get_figure()
        figure.savefig(f"{path}/{algorithm}.png")
        figure.clf()
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
