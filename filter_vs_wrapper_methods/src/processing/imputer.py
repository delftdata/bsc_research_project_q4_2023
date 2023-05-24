from typing import Literal, Union

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer


class Imputer:
    @staticmethod
    def impute_mean_or_median(df: pd.DataFrame, strategy: Literal["mean", "median"]) -> pd.DataFrame:
        imputer = SimpleImputer(missing_values=np.nan, strategy=strategy)
        imputer.set_output(transform="pandas")

        for i, column in enumerate(df.columns):
            if df[column].dtype != "float":
                continue
            if any(df.iloc[:, [i]].isna().values):
                df[column] = imputer.fit_transform(df.iloc[:, [i]])

        return df

    @staticmethod
    def impute_most_frequent(df: pd.DataFrame) -> pd.DataFrame:
        imputer = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
        imputer.set_output(transform="pandas")

        for i, column in enumerate(df.columns):
            if any(df.iloc[:, [i]].isna().values):
                df[column] = imputer.fit_transform(df.iloc[:, [i]])

        return df

    @staticmethod
    def impute_constant(df: pd.DataFrame, constant: Union[float, str, int, None] = 0) -> pd.DataFrame:
        return df.replace(np.nan, constant)

    @staticmethod
    def drop_missing_values(
            df: pd.DataFrame, axis: Literal[0, 1] = 0, how: Literal["any", "all"] = "any") -> pd.DataFrame:
        return df.dropna(axis=axis, how=how)
