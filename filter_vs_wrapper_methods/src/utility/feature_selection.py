from typing import Literal, Union

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectFdr, SequentialFeatureSelector
from sklearn.impute import SimpleImputer


class FeatureSelection:

    @staticmethod
    def split_input_target(df: pd.DataFrame, target_index: int) -> tuple[pd.DataFrame, pd.DataFrame]:
        input_indices = [j for j in range(df.shape[1]) if j != target_index]

        X = df.iloc[:, input_indices]
        y = df.iloc[:, [target_index]]

        return X, y

    @staticmethod
    def get_selected_features_indices(
            selector: SelectFdr or SequentialFeatureSelector,
            target_index: int) -> list[int]:

        selected_features_mask = selector.get_support()
        selected_features_indices = [target_index] + [i for (i, x) in enumerate(selected_features_mask) if x]
        return selected_features_indices

    @staticmethod
    def impute_missing_values(
            df: pd.DataFrame, missing_values: Union[int, float, str, None] = np.nan,
            strategy: Literal["mean", "median", "most_frequent", "constant", "drop"] = "drop") -> pd.DataFrame:

        if strategy == "drop":
            return df.dropna()

        imputer = SimpleImputer(missing_values, strategy)
        imputer.set_output(transform="pandas")
        df = imputer.fit_transform(df)
        return df
