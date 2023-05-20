from typing import Literal

import pandas as pd
from sklearn.feature_selection import chi2, f_classif
from utility.feature_selection import FeatureSelection


class Filter:
    def __init__(self, df: pd.DataFrame, method: Literal["Chi2", "ANOVA"], target_label: str):
        self.df = df
        self.method = method
        self.target_label = target_label
        X, y = FeatureSelection.split_input_target(self.df, self.target_label)

        if self.method == "Chi2":
            score_func = chi2
        else:
            score_func = f_classif

        print(f"Started filter feature selection, {self.method}.")
        statistic_value, _ = score_func(X, y)
        print(f"Finished filter feature selection.")

        sorted_indices = sorted([i for i in range(X.columns.size)],
                                key=lambda i: statistic_value[i], reverse=True)
        self.sorted_features = [str(column) for column in X.columns[sorted_indices]]

    def perform_feature_selection(self, selected_features_size=0.6) -> pd.DataFrame:
        k = int(selected_features_size * len(self.sorted_features))
        selected_k_features_names = self.sorted_features[0:k]

        selected_k_features_indices = [
            i for i in range(self.df.shape[1])
            if self.df.columns[i] in selected_k_features_names or self.df.columns[i] == self.target_label]

        return self.df.iloc[:, selected_k_features_indices]
