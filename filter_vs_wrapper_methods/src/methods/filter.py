from typing import Literal

import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from utility.feature_selection import FeatureSelection


class Filter:

    @staticmethod
    def perform_feature_selection(
            df: pd.DataFrame, target_index: int, filter_method: Literal["chi2", "anova"] = "chi2",
            selected_features_size=0.6) -> pd.DataFrame:
        X, y, df = FeatureSelection.split_input_target(df, target_index)

        print(f"Started filter feature selection, {filter_method}.")
        k_best_selector = SelectKBest(f_classif if filter_method == "anova" else chi2,
                                      k=int(selected_features_size * X.shape[1]))
        k_best_selector.fit(X, y)
        print(f"Finished filter feature selection.")
        selected_features_indices = FeatureSelection.get_selected_features_indices(k_best_selector)

        return df.iloc[:, selected_features_indices]

    # @staticmethod
    # def perform_optimized_feature_selection(df: pd.DataFrame, target_index: int) -> pd.DataFrame:
    #     pass
