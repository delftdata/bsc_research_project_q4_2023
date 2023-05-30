from typing import Literal

import pandas as pd
from processing.filter_preprocessing import preprocess_anova, preprocess_chi2
from processing.splitter import Splitter
from sklearn.feature_selection import chi2, f_classif


class Filter:

    @staticmethod
    def rank_features_descending(
            df: pd.DataFrame, method: Literal["chi2", "anova"],
            target_label: str, preprocessing=False) -> list[str]:

        preprocessed_df = df.copy()

        if preprocessing:
            if method == "chi2":
                preprocessed_df = preprocess_chi2(preprocessed_df, [target_label])
            else:
                preprocessed_df = preprocess_anova(preprocessed_df, [target_label])

        X, y = Splitter.split_input_target(preprocessed_df, target_label)
        score_func = chi2 if method == "chi2" else f_classif

        statistic_value = []

        try:
            print(f"Started filter feature selection, {method}.")
            statistic_value, _ = score_func(X, y)
            print(f"Finished filter feature selection.")
        except Exception as e:
            print(f"Finished filter feature selection with error, {method}: {e}.")

        sorted_indices = sorted([i for i in range(len(statistic_value))],
                                key=lambda i: statistic_value[i], reverse=True)

        sorted_features = [str(column) for column in X.columns[sorted_indices]]
        return sorted_features
