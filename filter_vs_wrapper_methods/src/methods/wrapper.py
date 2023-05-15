from typing import Literal

import pandas as pd
from sklearn.feature_selection import SequentialFeatureSelector
from utility.feature_selection import FeatureSelection


class Wrapper:

    @staticmethod
    def perform_feature_selection(df: pd.DataFrame, target_index: int, estimator,
                                  direction: Literal["forward", "backward"] = "forward", scoring="accuracy", tol=0.01,
                                  n_jobs=-1, n_features_to_select="auto") -> pd.DataFrame:
        X, y = FeatureSelection.split_input_target(df, target_index)

        sequential_selector = SequentialFeatureSelector(
            estimator, direction=direction, scoring=scoring, tol=tol, n_jobs=n_jobs,
            n_features_to_select=n_features_to_select)

        sequential_selector.fit(X, y)

        selected_features_indices = FeatureSelection.get_selected_features_indices(
            sequential_selector, target_index)

        return df.iloc[:, selected_features_indices]

    # @staticmethod
    # def perform_optimized_feature_selection(df: pd.DataFrame, target_index: int) -> pd.DataFrame:
    #     pass
