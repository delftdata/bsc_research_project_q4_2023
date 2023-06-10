import numpy as np
import pandas as pd
from sklearn.preprocessing import (KBinsDiscretizer, MinMaxScaler, Normalizer,
                                   OrdinalEncoder)


def scale_columns_min_max(df: pd.DataFrame, excluded_columns: list[str]) -> pd.DataFrame:
    """Scales the numerical columns in the DataFrame using Min-Max scaling.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.
    excluded_columns : list[str]
        List of column names to exclude from scaling.

    Returns
    -------
    pd.DataFrame
        The scaled DataFrame.
    """
    for i, column in enumerate(df.columns):
        if column not in excluded_columns:
            if df[column].dtype in set(["float", "int"]) and any(float(x) < 0 for x in df[column]):
                min_max_scaler = MinMaxScaler()
                df[column] = min_max_scaler.fit_transform(df.iloc[:, [i]])
    return df


def discretize_columns_k_bins(df: pd.DataFrame, excluded_columns: list[str]) -> pd.DataFrame:
    """Discretizes the float columns in the DataFrame using k-bins discretization.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.
    excluded_columns : list[str]
        List of column names to exclude from discretization.

    Returns
    -------
    pd.DataFrame
        The discretized DataFrame.
    """
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
    """Converts categorical columns in the DataFrame to numeric using ordinal encoding.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.
    excluded_columns : list[str]
        List of column names to exclude from encoding.

    Returns
    -------
    pd.DataFrame
        The encoded DataFrame.
    """
    for i, column in enumerate(df.columns):
        if column not in excluded_columns:
            if df[column].dtype != "float":
                ordinal_encoder = OrdinalEncoder(dtype=np.float64)
                df[column] = df[column].astype("category")
                df[column] = ordinal_encoder.fit_transform(df.iloc[:, [i]])
    return df


def normalize_df(df: pd.DataFrame, excluded_columns: list[str]) -> pd.DataFrame:
    """Normalizes the numerical columns in the DataFrame, therefore the provided DataFrame must contain only numerical
    features.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.
    excluded_columns : list[str]
        List of column names to exclude from normalization.

    Returns
    -------
    pd.DataFrame
        The normalized DataFrame.
    """
    normalizer = Normalizer()
    normalizer.set_output(transform="pandas")

    excluded_indices = [i for i, column in enumerate(df.columns) if column not in excluded_columns]
    df_included = df.iloc[:, excluded_indices]
    df_included = normalizer.fit_transform(df_included)

    for column in df_included.columns:  # type: ignore
        df[column] = df_included[column]

    return df


def drop_constant_feature(df: pd.DataFrame, excluded_columns: list[str]) -> pd.DataFrame:
    """Drops constant columns from the DataFrame. Columns are considered constant if they contain the same value for
    all rows.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.
    excluded_columns : list[str]
        List of column names to exclude from dropping.

    Returns
    -------
    pd.DataFrame
        The DataFrame with constant columns removed.
    """
    columns = df.columns

    for column in columns:
        if column not in excluded_columns and len(df[column].unique()) == 1:
            df.drop(column, inplace=True, axis=1)

    return df


def convert_to_actual_type(df: pd.DataFrame) -> pd.DataFrame:
    """Converts columns with inferred types to their actual data types.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.

    Returns
    -------
    pd.DataFrame
        The DataFrame with converted data types.
    """

    def is_float(value: str) -> bool:
        try:
            return not float(value).is_integer()
        except Exception:
            return False

    def is_int(value: str) -> bool:
        try:
            return float(value).is_integer()
        except Exception:
            return False

    for column in df.columns:
        if df[column].dtype == "object":
            if all(is_float(str(value)) for value in df[column].values):
                df[column] = df[column].astype("float64")
            elif all(is_int(str(value)) for value in df[column].values):
                df[column] = df[column].astype("int64")
            else:
                df[column] = df[column].astype("category")
    return df
