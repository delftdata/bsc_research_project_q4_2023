from __future__ import annotations

from typing import Literal

from numpy import nan
from pandas import DataFrame
from sklearn.impute import SimpleImputer


def impute_mean_or_median(df: DataFrame, strategy: Literal["mean", "median"]) -> DataFrame:
    """Imputes missing values in the DataFrame using the mean or median strategy.

    Parameters
    ----------
    df : DataFrame
        The input DataFrame.
    strategy : Literal["mean", "median"]
        The imputation strategy to use. Either "mean" or "median".

    Returns
    -------
    DataFrame
        The DataFrame with imputed values.
    """
    imputer = SimpleImputer(missing_values=nan, strategy=strategy)
    imputer.set_output(transform="pandas")

    for i, column in enumerate(df.columns):
        if df[column].dtype != "float":
            continue
        if any(df.iloc[:, [i]].isna().values):
            df[column] = imputer.fit_transform(df.iloc[:, [i]])

    return df


def impute_most_frequent(df: DataFrame) -> DataFrame:
    """Imputes missing values in the DataFrame using the most frequent strategy.

    Parameters
    ----------
    df : DataFrame
        The input DataFrame.

    Returns
    -------
    DataFrame
        The DataFrame with imputed values.
    """
    imputer = SimpleImputer(missing_values=nan, strategy="most_frequent")
    imputer.set_output(transform="pandas")

    for i, column in enumerate(df.columns):
        if any(df.iloc[:, [i]].isna().values):
            df[column] = imputer.fit_transform(df.iloc[:, [i]])

    return df


def impute_constant(df: DataFrame, constant: float | str | int | None = 0) -> DataFrame:
    """Imputes missing values in the DataFrame with a constant value.

    Parameters
    ----------
    df : DataFrame
        The input DataFrame.
    constant : float | str | int | None, optional
        The constant value to use for imputation (default: 0).

    Returns
    -------
    DataFrame
        The DataFrame with imputed values.
    """
    return df.replace(nan, constant)


def drop_missing_values(df: DataFrame, axis: Literal[0, 1] = 0, how: Literal["any", "all"] = "any") -> DataFrame:
    """Drops rows or columns with missing values from the DataFrame.

    Parameters
    ----------
    df : DataFrame
        The input DataFrame.
    axis : Literal[0, 1], optional
        The axis along which to drop the missing values. 0 for rows, 1 for columns (default: 0).
    how : Literal["any", "all"], optional
        The condition for dropping the values. "any" drops if any value is missing, "all" drops if all values are missing
        (default: "any").

    Returns
    -------
    DataFrame
        The DataFrame with missing values dropped.
    """
    return df.dropna(axis=axis, how=how)
