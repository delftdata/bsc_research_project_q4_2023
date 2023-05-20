from typing import Literal, Union

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (KBinsDiscretizer, MinMaxScaler,
                                   OrdinalEncoder)


class FeatureSelection:

    @staticmethod
    def split_input_target(df: pd.DataFrame, target_label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        target_index = df.columns.get_loc(key=target_label)

        X = df.iloc[:, [j for j in range(df.shape[1]) if j != target_index]]
        y = df.iloc[:, [target_index]]

        return X, y

    @staticmethod
    def impute_missing_values(
            df: pd.DataFrame,
            missing_values: Union[int, float, str, None] = np.nan,
            strategy: Literal["mean", "median", "most_frequent", "constant", "drop"] = "drop",
            constant: Union[str, float, int] = 0) -> pd.DataFrame:

        if strategy == "drop":
            return df.dropna()

        if strategy == "constant":
            return df.replace(missing_values, constant)

        imputer = SimpleImputer(missing_values=missing_values, strategy=strategy, fill_value=constant)
        imputer.set_output(transform="pandas")

        if strategy == "mean" or strategy == "median":
            for column in df.columns:
                if df[column].dtype == "float":
                    df[column] = imputer.fit_transform(df[[column]])
                if any([str(x).isnumeric() and not float(str(x)).is_integer() for x in df[column]]):
                    df[column] = df[column].astype("float64")
                    df[column] = imputer.fit_transform(df[[column]])

        if strategy == "most_frequent":
            for column in df.columns:
                df[column] = imputer.fit_transform(df[[column]])

        return df

    @staticmethod
    def preprocess_data(
            df: pd.DataFrame, feature_type: Literal["discrete", "continuous", "nominal", "ordinal"]) -> pd.DataFrame:

        if feature_type == "discrete":
            min_max_scaler = MinMaxScaler()
            for column in df.columns:
                if all([str(x).isnumeric() and float(str(x)).is_integer() for x in df[column]]):
                    df[column] = df[column].astype("int64")
                    df[column] = min_max_scaler.fit_transform(df[[column]])

        if feature_type == "continuous":
            k_bins_discretizer = KBinsDiscretizer(n_bins=df.shape[1], encode="ordinal", strategy="uniform")
            for column in df.columns:
                if any([str(x).isnumeric() and not float(str(x)).is_integer() for x in df[column]]) or \
                        df[column].dtype == "float":
                    df[column] = df[column].astype("float64")
                    df[column] = k_bins_discretizer.fit_transform(df[[column]])
                    df[column] = df[column].astype("int64").astype("category")

        if feature_type == "nominal" or feature_type == "ordinal":
            ordinal_encoder = OrdinalEncoder(dtype=np.float64)
            for column in df.columns:
                if df[column].dtype == "category" or df[column].dtype == "object" or any(
                        not str(x).isnumeric() for x in df[column]):
                    df[column] = df[column].astype("category")
                    df[column] = ordinal_encoder.fit_transform(df[[column]])
                    df[column] = df[column].astype("int64")

        return df
