from typing import Literal

import pandas as pd
from processing.wrapper_preprocessing import \
    preprocess_sequential_feature_selection
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LinearRegression, LogisticRegression
from splitting.splitter import Splitter


class Wrapper:

    @staticmethod
    def rank_features_descending(df: pd.DataFrame, method: Literal["forward_selection", "backward_elimination"],
                                 target_label: str, scoring: Literal["accuracy", "neg_mean_squared_error"],
                                 preprocessing=False, n_jobs=-1) -> list[str]:

        preprocessed_df = df.copy()

        if preprocessing:
            preprocessed_df = preprocess_sequential_feature_selection(preprocessed_df, [target_label])

        X, y = Splitter.split_input_target(preprocessed_df, target_label)

        if scoring == "accuracy":
            estimator = LogisticRegression()
        else:
            estimator = LinearRegression()

        if method == "forward_selection":
            sequential_selector = SequentialFeatureSelector(
                estimator=estimator, direction="forward", scoring=scoring, n_jobs=n_jobs,
                n_features_to_select=1)
        else:
            sequential_selector = SequentialFeatureSelector(
                estimator=estimator, direction="backward", scoring=scoring, n_jobs=n_jobs,
                n_features_to_select=X.columns.size-1)

        sorted_features: list[str] = []
        range_selection = [i for i in range(1, X.columns.size)]

        if method == "backward_elimination":
            range_selection.reverse()

        print(f"Started wrapper feature selection, {method}.")
        for i in range_selection:
            sequential_selector.set_params(n_features_to_select=i)
            sequential_selector.fit(X, y)
            for feature in sequential_selector.get_feature_names_out():
                if str(feature) not in sorted_features:
                    sorted_features.append(str(feature))
            print(f"Number of features: {i}")
        print(f"Finished wrapper feature selection.")

        sorted_features.append([column for column in X.columns if column not in sorted_features][0])
        if method == "backward_elimination":
            sorted_features.reverse()

        return sorted_features
