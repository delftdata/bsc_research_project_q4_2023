from time import perf_counter
from typing import Literal

import pandas as pd
from numpy import nan
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LinearRegression, LogisticRegression

from processing.splitter import split_input_target
from processing.wrapper_preprocessing import \
    preprocess_sequential_feature_selection


def rank_features_descending_wrapper(df: pd.DataFrame, method: Literal["forward_selection", "backward_elimination"],
                                     target_label: str, scoring: Literal["accuracy", "neg_root_mean_squared_error"],
                                     preprocessing=False, normalization=True, n_jobs=-1) -> tuple[list[str], float]:
    """Ranks features in descending order using Forward Selection or Backward Elimination.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the dataset.
    method : Literal["forward_selection", "backward_elimination"]
        The wrapper feature selection technique to be used for feature ranking.
    target_label : str
        The target label column in the DataFrame.
    scoring : Literal["accuracy", "neg_root_mean_squared_error"]
        The scoring metric to be used for feature evaluation.
    preprocessing : bool, optional
        Flag indicating whether to perform preprocessing on the data before ranking features (default: False).
    normalization : bool, optional
        Flag indicating whether to perform feature normalization during preprocessing (default: True).
    n_jobs : int, optional
        The number of parallel jobs to run during feature selection. Default is -1, which uses all available processors.

    Returns
    -------
    tuple[list[str], float]
        A tuple containing the sorted feature names in descending order and the runtime of the feature ranking.
    """

    preprocessed_df = df.copy()

    if preprocessing:
        preprocessed_df = preprocess_sequential_feature_selection(preprocessed_df, [target_label], normalization)

    X, y = split_input_target(preprocessed_df, target_label)
    estimator = LogisticRegression() if scoring == "accuracy" else LinearRegression()
    sequential_selector = SequentialFeatureSelector(
        estimator=estimator, direction="forward" if method == "forward_selection" else "backward", scoring=scoring,
        n_jobs=n_jobs, n_features_to_select=1)

    sorted_features: list[str] = []
    runtime = nan
    range_selection = list(range(1, X.columns.size))
    if method == "backward_elimination":
        range_selection.reverse()

    try:
        sorted_features, runtime = perform_wrapper_feature_selection(range_selection, sequential_selector, X, y, method)
    except Exception as error:
        print(f"Finished wrapper feature selection with error, {method}: {error}.")

    sorted_features.append([column for column in X.columns if column not in sorted_features][0])
    if method == "backward_elimination":
        sorted_features.reverse()
    return sorted_features, runtime


def perform_wrapper_feature_selection(range_selection: list[int],
                                      sequential_selector: SequentialFeatureSelector, X: pd.DataFrame, y,
                                      method: Literal["forward_selection", "backward_elimination"]
                                      ) -> tuple[list[str], float]:
    """Performs feature selection using a wrapper method, either Forward Selection or Backward Elimination.

    Parameters
    ----------
    range_selection : list[int]
        A list specifying the range of feature selections to iterate over.
        Forward Selection: [1, number of columns of X]
        Backward Elimination: [number of columns of X, 1]
    sequential_selector : SequentialFeatureSelector
        The sequential feature selector object.
    X : pd.DataFrame
        The feature matrix.
    y : ndarray or Series
        The target variable.
    method : Literal["forward_selection", "backward_elimination"]
        The wrapper method for feature selection.

    Returns
    -------
    tuple[list[str], float]
        A tuple containing the sorted feature names and the runtime of the feature selection process.
    """
    sorted_features: list[str] = []
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

    end = perf_counter()
    runtime = end - start
    return sorted_features, runtime
