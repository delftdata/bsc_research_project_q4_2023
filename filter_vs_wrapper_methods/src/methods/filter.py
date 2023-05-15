from typing import Literal

import pandas as pd
from sklearn.feature_selection import SelectFdr, chi2, f_classif
from utility.feature_selection import FeatureSelection


class Filter:

    @staticmethod
    def perform_feature_selection(
            df: pd.DataFrame, target_index: int, filter_method: Literal["chi2", "anova"] = "chi2",
            significance_level=0.05) -> pd.DataFrame:
        X, y, df = FeatureSelection.split_input_target(df, target_index)

        print(f"Started filter feature selection.")
        false_discovery_rate = SelectFdr(f_classif if filter_method == "anova" else chi2,
                                         alpha=significance_level)
        false_discovery_rate.fit(X, y)
        print(f"Finished filter feature selection.")
        selected_features_indices = FeatureSelection.get_selected_features_indices(false_discovery_rate)

        return df.iloc[:, selected_features_indices]

    # @staticmethod
    # def perform_optimized_feature_selection(df: pd.DataFrame, target_index: int) -> pd.DataFrame:
    #     pass
