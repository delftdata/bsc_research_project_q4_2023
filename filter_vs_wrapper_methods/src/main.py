import os
import warnings

import numpy as np
import pandas as pd
from methods.filter import Filter
from methods.wrapper import Wrapper
from sklearn.calibration import LinearSVC
from utility.feature_selection import FeatureSelection

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"


def perform_feature_selection_on_arrhythmia_dataset():
    file_dataset = "data/arrhythmia/arrhythmia.data"
    file_labels = "data/arrhythmia/labels.names"

    df = pd.read_csv(file_dataset, header=None, low_memory=False)
    df_column_names = pd.read_csv(file_labels, header=None)

    df = df.rename(columns=dict([(i, df_column_names[0][i]) for i in range(df.shape[1])]))
    df = df.replace("?", np.nan)
    df = FeatureSelection.impute_missing_values(df)

    chi2, anova, forward_selection, backward_elimination = None, None, None, None

    try:
        chi2 = Filter.perform_feature_selection(df, df.shape[1] - 1, "chi2")
    except Exception as e:
        print("Chi2:", e)

    try:
        anova = Filter.perform_feature_selection(df, df.shape[1] - 1, "anova")
    except Exception as e:
        print("Anova:", e)

    try:
        forward_selection = Wrapper.perform_feature_selection(df, df.shape[1] - 1, LinearSVC(), direction="forward")
    except Exception as e:
        print("Forward Selection: ", e)

    try:
        backward_elimination = Wrapper.perform_feature_selection(df, df.shape[1] - 1, LinearSVC(), direction="backward")
    except Exception as e:
        print("Forward Selection: ", e)

    return chi2, anova, forward_selection, backward_elimination


if __name__ == "__main__":
    chi2, anova, forward_selection, backward_elimination = perform_feature_selection_on_arrhythmia_dataset()
    print("Chi2:", chi2.shape if chi2 is not None else None)
    print("ANOVA:", anova.shape if anova is not None else None)
    print("Forward Selection:", forward_selection.shape if forward_selection is not None else None)
    print("Backward Elimination:", backward_elimination.shape if backward_elimination is not None else None)
