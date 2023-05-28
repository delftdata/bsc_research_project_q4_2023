from time import perf_counter
from typing import Literal

import pandas as pd
from numpy import nan
from processing.splitter import split_input_target
from processing.wrapper_preprocessing import \
    preprocess_sequential_feature_selection
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LinearRegression, LogisticRegression


def rank_features_descending_wrapper(df: pd.DataFrame, method: Literal["forward_selection", "backward_elimination"],
                                     target_label: str, scoring: Literal["accuracy", "neg_root_mean_squared_error"],
                                     preprocessing=False, n_jobs=-1) -> tuple[list[str], float]:

    preprocessed_df = df.copy()

    if preprocessing:
        preprocessed_df = preprocess_sequential_feature_selection(preprocessed_df, [target_label])

    X, y = split_input_target(preprocessed_df, target_label)
    estimator = LogisticRegression() if scoring == "accuracy" else LinearRegression()
    sequential_selector = SequentialFeatureSelector(
        estimator=estimator, direction="forward" if method == "forward_selection" else "backward", scoring=scoring,
        n_jobs=n_jobs, n_features_to_select=1)

    sorted_features: list[str] = []
    runtime = nan
    range_selection = [i for i in range(1, X.columns.size)]
    if method == "backward_elimination":
        range_selection.reverse()

    try:
        # print(f"Started wrapper feature selection, {method}.")
        start = perf_counter()
        for i in range_selection:
            sequential_selector.set_params(n_features_to_select=i)
            sequential_selector.fit(X, y)

            if method == "forward_selection":
                for feature in sequential_selector.get_feature_names_out():
                    if str(feature) not in sorted_features:
                        sorted_features.append(str(feature))
            else:
                for feature in X.columns:
                    if str(feature) not in sequential_selector.get_feature_names_out() and \
                            str(feature) not in sorted_features:
                        sorted_features.append(str(feature))

            # print(f"Number of features: {len(sorted_features)}")
        end = perf_counter()
        # print(f"Finished wrapper feature selection, {method}.")

        runtime = end - start
    except Exception as e:
        print(f"Finished wrapper feature selection with error, {method}: {e}.")

    sorted_features.append([column for column in X.columns if column not in sorted_features][0])
    if method == "backward_elimination":
        sorted_features.reverse()
    return sorted_features, runtime
