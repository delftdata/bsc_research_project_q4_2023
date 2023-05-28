from time import perf_counter
from typing import Literal

import pandas as pd
from numpy import nan
from processing.filter_preprocessing import preprocess_anova, preprocess_chi2
from processing.splitter import split_input_target
from sklearn.feature_selection import chi2, f_classif


def rank_features_descending_filter(df: pd.DataFrame, method: Literal["chi2", "anova"],
                                    target_label: str, preprocessing=False) -> tuple[list[str], float]:

    preprocessed_df = df.copy()

    if preprocessing:
        if method == "chi2":
            preprocessed_df = preprocess_chi2(preprocessed_df, [target_label])
        else:
            preprocessed_df = preprocess_anova(preprocessed_df, [target_label])

    X, y = split_input_target(preprocessed_df, target_label)
    score_func = chi2 if method == "chi2" else f_classif
    statistic_value = []
    runtime = nan

    try:
        # print(f"Started filter feature selection, {method}.")
        start = perf_counter()
        statistic_value, _ = score_func(X, y)
        end = perf_counter()
        # print(f"Finished filter feature selection.")
        runtime = end - start
    except Exception as e:
        print(f"Finished filter feature selection with error, {method}: {e}.")

    sorted_indices = sorted([i for i, _ in enumerate(statistic_value)],
                            key=lambda i: statistic_value[i], reverse=True)

    sorted_features = [str(column) for column in X.columns[sorted_indices]]
    return sorted_features, runtime
