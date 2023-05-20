from typing import Literal

import pandas as pd
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LinearRegression, LogisticRegression
from utility.feature_selection import FeatureSelection


class Wrapper:
    def __init__(
            self,
            df: pd.DataFrame, method: Literal["Forward Selection", "Backward Elimination"],
            target_label: str, scoring: Literal["accuracy", "neg_mean_squared_error"],
            n_jobs=-1):
        self.df = df
        self.method = method
        self.target_label = target_label
        self.scoring = scoring

        X, y = FeatureSelection.split_input_target(self.df, self.target_label)

        if self.scoring == "accuracy":
            estimator = LogisticRegression()
        else:
            estimator = LinearRegression()

        if self.method == "Forward Selection":
            sequential_selector = SequentialFeatureSelector(
                estimator=estimator, direction="forward", scoring=self.scoring, n_jobs=n_jobs,
                n_features_to_select=1)
        else:
            sequential_selector = SequentialFeatureSelector(
                estimator=estimator, direction="backward", scoring=self.scoring, n_jobs=n_jobs,
                n_features_to_select=X.columns.size-1)

        self.sorted_features: list[str] = []

        range_selection = [i for i in range(1, X.columns.size)]
        if self.method == "Backward Elimination":
            range_selection.reverse()

        print(f"Started wrapper feature selection, {method}.")
        for i in range_selection:
            sequential_selector.set_params(n_features_to_select=i)
            sequential_selector.fit(X, y)
            for feature in sequential_selector.get_feature_names_out():
                if str(feature) not in self.sorted_features:
                    self.sorted_features.append(str(feature))
            print(f"Number of features: {i}")
        print(f"Finished wrapper feature selection.")

        self.sorted_features.append([column for column in X.columns if column not in self.sorted_features][0])
        if self.method == "Backward Elimination":
            self.sorted_features.reverse()

    def perform_feature_selection(self, selected_features_size=0.6) -> pd.DataFrame:
        k = int(selected_features_size * len(self.sorted_features))
        selected_k_features_names = self.sorted_features[0:k]

        selected_k_features_indices = [
            i for i in range(self.df.shape[1])
            if self.df.columns[i] in selected_k_features_names or self.df.columns[i] == self.target_label]

        return self.df.iloc[:, selected_k_features_indices]
