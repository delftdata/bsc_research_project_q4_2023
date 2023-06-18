from __future__ import annotations

from typing import Literal

import pandas as pd
from numpy import nan

from processing.imputer import (drop_missing_values, impute_mean_or_median,
                                impute_most_frequent)
from processing.preprocessing import convert_to_actual_type
from reader.dataset_info import DatasetInfo


class Reader:
    """A class used to read and preprocess data for experiments by imputing or dropping missing values.

    Attributes
    ----------
    experiment_name : str
        Name of the experiment (valid experiment names: experiment1, experiment2, experiment3, experiment4).
    imputation_strategy : Literal["mean", "median"]
        Name of the imputation strategy used to handle missing values (default: "mean").
    """

    def __init__(self, experiment_name: str, imputation_strategy: Literal["mean", "median"] = "mean"):
        """
        Parameters
        ----------
        experiment_name : str
            Name of the experiment (valid experiment names: experiment1, experiment2, experiment3, experiment4).
        imputation_strategy : Literal["mean", "median"], optional
            Name of the imputation strategy used to handle missing values (default: "mean").
        """
        self.experiment_name = experiment_name
        self.imputation_strategy: Literal["mean", "median"] = imputation_strategy

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

    def read_bank_marketing(self) -> tuple[pd.DataFrame, DatasetInfo]:
        """Reads the bank marketing dataset from a CSV file and prepares the DataFrame.

        Returns
        -------
        tuple[pd.DataFrame, DatasetInfo]
            A tuple containing the prepared DataFrame and the DatasetInfo object.

        Raises
        ------
        FileNotFoundError
            If the bank marketing dataset file is not found.
        """
        bank = DatasetInfo("bank_marketing", "reader/data/bank_marketing/bank.csv",
                           "y", f"results/{self.experiment_name}/bank_marketing")
        df_bank = pd.read_csv(bank.dataset_path, low_memory=False)
        df_bank = df_bank.replace("unknown", nan)
        df_bank = self.prepare_data_frame(df=df_bank, missing_values=True)
        return df_bank, bank

    def read_breast_cancer(self) -> tuple[pd.DataFrame, DatasetInfo]:
        """Reads the breast cancer dataset from a CSV file and prepares the DataFrame.

        Returns
        -------
        tuple[pd.DataFrame, DatasetInfo]
            A tuple containing the prepared DataFrame and the DatasetInfo object.

        Raises
        ------
        FileNotFoundError
            If the breast cancer dataset file is not found.
        """
        breast_cancer = DatasetInfo("breast_cancer", "reader/data/breast_cancer/breast_cancer.csv", "diagnosis",
                                    f"results/{self.experiment_name}/breast_cancer")
        df_breast_cancer = pd.read_csv(breast_cancer.dataset_path, low_memory=False)
        df_breast_cancer = self.prepare_data_frame(df=df_breast_cancer)
        return df_breast_cancer, breast_cancer

    def read_steel_plates_faults(self) -> tuple[pd.DataFrame, DatasetInfo]:
        """Reads the steel plates faults dataset from a CSV file and prepares the DataFrame.

        Returns
        -------
        tuple[pd.DataFrame, DatasetInfo]
            A tuple containing the prepared DataFrame and the DatasetInfo object.

        Raises
        ------
        FileNotFoundError
            If the steel plates faults dataset file is not found.
        """
        steel_plates_faults = DatasetInfo(
            "steel_plates_faults", "reader/data/steel_plates_faults/steel_plates_faults.csv", "Class",
            f"results/{self.experiment_name}/steel_plates_faults")
        df_steel_plates_faults = pd.read_csv(steel_plates_faults.dataset_path, low_memory=False)
        df_steel_plates_faults = self.prepare_data_frame(df=df_steel_plates_faults)
        return df_steel_plates_faults, steel_plates_faults

    def read_housing_prices(self) -> tuple[pd.DataFrame, DatasetInfo]:
        """Reads the housing prices dataset from a CSV file and prepares the DataFrame.

        Returns
        -------
        tuple[pd.DataFrame, DatasetInfo]
            A tuple containing the prepared DataFrame and the DatasetInfo object.

        Raises
        ------
        FileNotFoundError
            If the housing prices dataset file is not found.
        """
        housing_prices = DatasetInfo(
            "housing_prices", "reader/data/housing_prices/housing_prices.csv", "SalePrice",
            f"results/{self.experiment_name}/housing_prices", eval_metric="neg_root_mean_squared_error")
        df_housing_prices = pd.read_csv(housing_prices.dataset_path, low_memory=False)
        df_housing_prices = df_housing_prices.fillna(nan)
        df_housing_prices = self.prepare_data_frame(df=df_housing_prices, missing_values=True)
        return df_housing_prices, housing_prices

    def read_bike_sharing(self) -> tuple[pd.DataFrame, DatasetInfo]:
        """Reads the bike sharing dataset from a CSV file and prepares the DataFrame.

        Returns
        -------
        tuple[pd.DataFrame, DatasetInfo]
            A tuple containing the prepared DataFrame and the DatasetInfo object.

        Raises
        ------
        FileNotFoundError
            If the bike sharing dataset file is not found.
        """
        bike_sharing = DatasetInfo("bike_sharing", "reader/data/bike_sharing/hour.csv", "cnt",
                                   f"results/{self.experiment_name}/bike_sharing",
                                   eval_metric="neg_root_mean_squared_error")
        df_bike_sharing = pd.read_csv(bike_sharing.dataset_path, low_memory=False)
        df_bike_sharing = self.prepare_data_frame(df=df_bike_sharing)
        return df_bike_sharing, bike_sharing

    def read_census_income(self) -> tuple[pd.DataFrame, DatasetInfo]:
        """Reads the census income dataset from a CSV file and prepares the DataFrame.

        Returns
        -------
        tuple[pd.DataFrame, DatasetInfo]
            A tuple containing the prepared DataFrame and the DatasetInfo object.

        Raises
        ------
        FileNotFoundError
            If the census income dataset file is not found.
        """
        census_income = DatasetInfo("census_income", "reader/data/census_income/census_income.csv", "income_label",
                                    f"results/{self.experiment_name}/census_income")
        df_census_income = pd.read_csv(census_income.dataset_path, low_memory=False)
        df_census_income = df_census_income.fillna(nan)
        df_census_income = self.prepare_data_frame(df=df_census_income, missing_values=True)
        return df_census_income, census_income

    def read_nasa_numeric(self) -> tuple[pd.DataFrame, DatasetInfo]:
        """Reads the nasa numeric dataset from a CSV file and prepares the DataFrame.

        Returns
        -------
        tuple[pd.DataFrame, DatasetInfo]
            A tuple containing the prepared DataFrame and the DatasetInfo object.

        Raises
        ------
        FileNotFoundError
            If the nasa numeric dataset file is not found.
        """
        nasa_numeric = DatasetInfo(
            "nasa_numeric", "reader/data/nasa_numeric/nasa_numeric.csv", "act_effort",
            f"results/{self.experiment_name}/nasa_numeric", eval_metric="neg_root_mean_squared_error")
        df_nasa_numeric = pd.read_csv(nasa_numeric.dataset_path, low_memory=False)
        df_nasa_numeric = df_nasa_numeric.fillna(nan)
        df_nasa_numeric = self.prepare_data_frame(df=df_nasa_numeric, missing_values=True)
        return df_nasa_numeric, nasa_numeric

    def read_arrhythmia(self) -> tuple[pd.DataFrame, DatasetInfo]:
        """Reads the arrhythmia dataset from a CSV file and prepares the DataFrame.

        Returns
        -------
        tuple[pd.DataFrame, DatasetInfo]
            A tuple containing the prepared DataFrame and the DatasetInfo object.

        Raises
        ------
        FileNotFoundError
            If the arrhythmia dataset file is not found.
        """
        arrhythmia = DatasetInfo("arrhythmia", "reader/data/arrhythmia/arrhythmia.csv", "Class",
                                 f"results/{self.experiment_name}/arrhythmia")
        df_arrhythmia = pd.read_csv(arrhythmia.dataset_path, low_memory=False)
        df_arrhythmia = df_arrhythmia.replace("?", nan)
        df_arrhythmia = self.prepare_data_frame(df=df_arrhythmia, missing_values=True)
        return df_arrhythmia, arrhythmia

    def read_crop(self) -> tuple[pd.DataFrame, DatasetInfo]:
        """Reads the crop dataset from multiple CSV files and concatenates them into a single DataFrame.

        Returns
        -------
        tuple[pd.DataFrame, DatasetInfo]
            A tuple containing the concatenated DataFrame and the DatasetInfo object.

        Raises
        ------
        FileNotFoundError
            If any of the crop dataset files are not found.
        """
        crop = DatasetInfo("crop", "reader/data/crop", "label", f"results/{self.experiment_name}/crop")
        frames = []
        for i in range(2):
            frames.append(pd.read_csv(f"{crop.dataset_path}/crop{i}.csv", low_memory=False))
        df_crop = pd.concat(frames)
        df_crop = self.prepare_data_frame(df=df_crop)
        return df_crop, crop

    def read_character_font_images(self) -> tuple[pd.DataFrame, DatasetInfo]:
        """Reads the character font images dataset from multiple CSV files and concatenates them into a single DataFrame.

        Returns
        -------
        tuple[pd.DataFrame, DatasetInfo]
            A tuple containing the concatenated DataFrame and the DatasetInfo object.

        Raises
        ------
        FileNotFoundError
            If any of the character font images dataset files or the font names file are not found.
        """
        character_font_images = DatasetInfo(
            "character_font_images", "reader/data/character_font_images", "font",
            f"results/{self.experiment_name}/character_font_images",
            file_names="reader/data/character_font_images/font.names")
        df_file_names = pd.read_csv(character_font_images.file_names, low_memory=False, header=None)
        frames = []
        for i, file_name in enumerate(df_file_names[0]):
            frames.append(pd.read_csv(f"{character_font_images.dataset_path}/{file_name}"))
            if i == 2:
                break
        df_character_font_images = pd.concat(frames)
        df_character_font_images = self.prepare_data_frame(df=df_character_font_images)
        return df_character_font_images, character_font_images

    def read_internet_advertisements(self) -> tuple[pd.DataFrame, DatasetInfo]:
        """Reads the internet advertisements dataset from a CSV file.

        Returns
        -------
        tuple[pd.DataFrame, DatasetInfo]
            A tuple containing the DataFrame and the DatasetInfo object.

        Raises
        ------
        FileNotFoundError
            If the internet advertisements dataset file is not found.
        """
        internet_ads = DatasetInfo("internet_advertisements",
                                   "reader/data/internet_advertisements/internet_advertisements.csv", "class",
                                   f"results/{self.experiment_name}/internet_advertisements")
        df_internet_ads = pd.read_csv(internet_ads.dataset_path, low_memory=False)
        df_internet_ads = df_internet_ads.replace("   ?", nan)
        df_internet_ads = df_internet_ads.replace("     ?", nan)
        df_internet_ads = df_internet_ads.replace("?", nan)
        df_internet_ads = self.prepare_data_frame(df=df_internet_ads, missing_values=True)
        return df_internet_ads, internet_ads
