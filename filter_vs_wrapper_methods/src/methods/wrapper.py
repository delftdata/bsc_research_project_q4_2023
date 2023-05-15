from typing import Literal

import pandas as pd
from sklearn.feature_selection import SequentialFeatureSelector
from utility.feature_selection import FeatureSelection


class Wrapper:

    @staticmethod
    def perform_feature_selection(df: pd.DataFrame, target_index: int, estimator,
                                  direction: Literal["forward", "backward"] = "forward", scoring="accuracy",
                                  n_jobs=-1, n_features_to_select=0.6) -> pd.DataFrame:
        X, y, df = FeatureSelection.split_input_target(df, target_index)

        print(f"Started wrapper feature selection, direction: {direction}.")
        sequential_selector = SequentialFeatureSelector(
            estimator, direction=direction, scoring=scoring, n_jobs=n_jobs,
            n_features_to_select=n_features_to_select)
        sequential_selector.fit(X, y)
        print(f"Finished wrapper feature selection.")
        selected_features_indices = FeatureSelection.get_selected_features_indices(sequential_selector)

        return df.iloc[:, selected_features_indices]

    # @staticmethod
    # def perform_optimized_feature_selection(df: pd.DataFrame, target_index: int) -> pd.DataFrame:
    #     pass
