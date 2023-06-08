from time import perf_counter
from typing import Literal

import pandas as pd
from numpy import nan
from sklearn.feature_selection import chi2, f_classif

from processing.filter_preprocessing import preprocess_anova, preprocess_chi2
from processing.splitter import split_input_target


def rank_features_descending_filter(df: pd.DataFrame, method: Literal["chi2", "anova"],
                                    target_label: str, preprocessing=False, normalization=True) -> tuple[list[str], float]:
    """Ranks features in descending order based on the Chi-Squared or ANOVA test.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the dataset.
    method : Literal["chi2", "anova"]
        The filter feature selection technique to be used for feature ranking.
    target_label : str
        The target label column in the DataFrame.
    preprocessing : bool, optional
        Flag indicating whether to perform preprocessing on the data before ranking features (default: False)
    normalization : bool, optional
        Flag indicating whether to perform feature normalization during preprocessing (default: True).

    Returns
    -------
    tuple[list[str], float]
        A tuple containing the sorted feature names in descending order and the runtime of the feature ranking.
    """

    preprocessed_df = df.copy()

    if preprocessing:
        if method == "chi2":
            preprocessed_df = preprocess_chi2(preprocessed_df, [target_label])
        else:
            preprocessed_df = preprocess_anova(preprocessed_df, [target_label], normalization)

    X, y = split_input_target(preprocessed_df, target_label)
    score_func = chi2 if method == "chi2" else f_classif
    statistic_value = []
    runtime = nan

    try:
        # print(f"Started filter feature selection, {method}.")
        start = perf_counter()
        statistic_value, _ = score_func(X, y)
        end = perf_counter()
        # print(f"Finished filter feature selection, {method}.")
        runtime = end - start
    except Exception as e:
        print(f"Finished filter feature selection with error, {method}: {e}.")

    sorted_indices = sorted([i for i, _ in enumerate(statistic_value)],
                            key=lambda i: statistic_value[i], reverse=True)

    sorted_features = [str(column) for column in X.columns[sorted_indices]]
    return sorted_features, runtime
