from typing import Literal, Union

import pandas as pd
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.svm import LinearSVC, LinearSVR
from utility.feature_selection import FeatureSelection


class Wrapper:

    @staticmethod
    def perform_feature_selection(
            df: pd.DataFrame, target_index: int, estimator: Union[LinearSVC, LinearSVR, SGDClassifier, SGDRegressor],
            scoring: Literal["accuracy", "balanced_accuracy", "f1", "precision", "recall", "jaccard", "roc_auc",
                             "explained_variance", "max_error", "neg_mean_absolute_error", "r2"],
            direction: Literal["forward", "backward"] = "forward",
            n_jobs=-1, selected_features_size=0.6) -> pd.DataFrame:
        X, y, df = FeatureSelection.split_input_target(df, target_index)

        print(f"Started wrapper feature selection, direction: {direction}.")
        sequential_selector = SequentialFeatureSelector(
            estimator, direction=direction, scoring=scoring, n_jobs=n_jobs,
            n_features_to_select=selected_features_size)
        sequential_selector.fit(X, y)
        print(f"Finished wrapper feature selection.")
        selected_features_indices = FeatureSelection.get_selected_features_indices(sequential_selector)

        return df.iloc[:, selected_features_indices]

    # @staticmethod
    # def perform_optimized_feature_selection(df: pd.DataFrame, target_index: int) -> pd.DataFrame:
    #     pass
