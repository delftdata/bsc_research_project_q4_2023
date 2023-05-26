import numpy as np
import pandas as pd
from sklearn.preprocessing import (KBinsDiscretizer, MinMaxScaler, Normalizer,
                                   OrdinalEncoder)


def scale_columns_min_max(df: pd.DataFrame, excluded_columns: list[str]) -> pd.DataFrame:
    for i, column in enumerate(df.columns):
        if column not in excluded_columns:
            if (df[column].dtype == "float" or df[column].dtype == "int") and any([float(x) < 0 for x in df[column]]):
                min_max_scaler = MinMaxScaler()
                df[column] = min_max_scaler.fit_transform(df.iloc[:, [i]])
    return df


def discretize_columns_k_bins(df: pd.DataFrame, excluded_columns: list[str]) -> pd.DataFrame:
    for i, column in enumerate(df.columns):
        if column not in excluded_columns:
            if df[column].dtype == "float" and all(float(x).is_integer() for x in df[column]):
                df[column] = df[column].astype("int64")
            elif df[column].dtype == "float":
                k_bins_discretizer = KBinsDiscretizer(n_bins=df.shape[1], encode="ordinal", strategy="uniform")
                df[column] = k_bins_discretizer.fit_transform(df.iloc[:, [i]])
                df[column] = df[column].astype("int64")
    return df


def discretize_columns_ordinal_encoder(df: pd.DataFrame, excluded_columns: list[str]) -> pd.DataFrame:
    for i, column in enumerate(df.columns):
        if column not in excluded_columns:
            if df[column].dtype != "float":
                ordinal_encoder = OrdinalEncoder(dtype=np.float64)
                df[column] = df[column].astype("category")
                df[column] = ordinal_encoder.fit_transform(df.iloc[:, [i]])
    return df


def normalize_df(df: pd.DataFrame, excluded_columns: list[str]) -> pd.DataFrame:
    normalizer = Normalizer()
    normalizer.set_output(transform="pandas")

    excluded_indices = [i for i, column in enumerate(df.columns) if column not in excluded_columns]
    df_included = df.iloc[:, excluded_indices]
    df_included = normalizer.fit_transform(df_included)

    for column in df_included.columns:  # type: ignore
        df[column] = df_included[column]

    return df


def drop_constant_feature(df: pd.DataFrame, excluded_columns: list[str]) -> pd.DataFrame:
    columns = df.columns

    for column in columns:
        if column not in excluded_columns and len(df[column].unique()) == 1:
            df.drop(column, inplace=True, axis=1)

    return df


def convert_to_actual_type(df: pd.DataFrame) -> pd.DataFrame:
    def is_float(value: str) -> bool:
        try:
            float(value)
            return True
        except:
            return False

    for column in df.columns:
        if df[column].dtype == "object":
            if all([is_float(str(value)) for value in df[column].values]):
                df[column] = df[column].astype("float64")
            elif all([str(value).isnumeric() for value in df[column].values]):
                df[column] = df[column].astype("int64")
            else:
                df[column] = df[column].astype("category")
    return df
