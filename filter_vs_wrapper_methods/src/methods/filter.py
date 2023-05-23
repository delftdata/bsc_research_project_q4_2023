from typing import Literal

import pandas as pd
from preprocessing.filter_preprocessing import (preprocess_anova,
                                                preprocess_chi2)
from sklearn.feature_selection import chi2, f_classif
from splitting.splitter import Splitter


class Filter:
    def __init__(self, df: pd.DataFrame, method: Literal["Chi2", "ANOVA"], target_label: str, preprocessing=False):
        preprocessed_df = df.copy()
        self.method = method
        self.target_label = target_label

        if preprocessing:
            if self.method == "Chi2":
                preprocessed_df = preprocess_chi2(preprocessed_df, [self.target_label])
            else:
                preprocessed_df = preprocess_anova(preprocessed_df, [self.target_label])

        X, y = Splitter.split_input_target(preprocessed_df, self.target_label)

        if self.method == "Chi2":
            score_func = chi2
        else:
            score_func = f_classif

        statistic_value = []

        try:
            print(f"Started filter feature selection, {self.method}.")
            statistic_value, _ = score_func(X, y)
            print(f"Finished filter feature selection.")
        except Exception as e:
            print(f"Finished filter feature selection with error, {self.method}: {e}.")

        sorted_indices = sorted([i for i in range(len(statistic_value))],
                                key=lambda i: statistic_value[i], reverse=True)
        self.sorted_features = [str(column) for column in X.columns[sorted_indices]]

    def perform_feature_selection(self, df: pd.DataFrame, selected_features_size=0.6) -> pd.DataFrame:
        k = int(selected_features_size * len(self.sorted_features))
        selected_k_features_names = self.sorted_features[0:k]

        selected_k_features_indices = [
            i for i in range(df.shape[1])
            if df.columns[i] in selected_k_features_names or df.columns[i] == self.target_label]

        return df.iloc[:, selected_k_features_indices]
