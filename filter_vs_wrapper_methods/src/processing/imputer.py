from typing import Literal, Union

from numpy import nan
from pandas import DataFrame
from sklearn.impute import SimpleImputer


def impute_mean_or_median(df: DataFrame, strategy: Literal["mean", "median"]) -> DataFrame:
    imputer = SimpleImputer(missing_values=nan, strategy=strategy)
    imputer.set_output(transform="pandas")

    for i, column in enumerate(df.columns):
        if df[column].dtype != "float":
            continue
        if any(df.iloc[:, [i]].isna().values):
            df[column] = imputer.fit_transform(df.iloc[:, [i]])

    return df


def impute_most_frequent(df: DataFrame) -> DataFrame:
    imputer = SimpleImputer(missing_values=nan, strategy="most_frequent")
    imputer.set_output(transform="pandas")

    for i, column in enumerate(df.columns):
        if any(df.iloc[:, [i]].isna().values):
            df[column] = imputer.fit_transform(df.iloc[:, [i]])

    return df


def impute_constant(df: DataFrame, constant: Union[float, str, int, None] = 0) -> DataFrame:
    return df.replace(nan, constant)


def drop_missing_values(df: DataFrame, axis: Literal[0, 1] = 0, how: Literal["any", "all"] = "any") -> DataFrame:
    return df.dropna(axis=axis, how=how)
