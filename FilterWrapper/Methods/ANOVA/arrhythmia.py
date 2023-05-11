import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectFdr
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import StandardScaler


def arrhythmia_anova():
    df = pd.read_csv("/FilterWrapper/Datasets/Arrhythmia/arrhythmia.data",
                     header=None, low_memory=False)

    df_column_names = pd.read_csv("/FilterWrapper/Datasets/Arrhythmia/labels.names", header=None)

    df.rename(columns=dict([(i, df_column_names[0][i]) for i in range(df.shape[1])]), inplace=True)

    for i in [2, 22, 23, 24, 25, 26, 27, 34, 35, 36, 37, 38, 39, 46, 47,
              48, 49, 50, 51, 58, 59, 60, 61, 62, 63, 70, 71, 72, 73, 74, 75, 82, 83, 84, 85, 86, 87,
              94, 95, 96, 97, 98, 99, 106, 107, 108, 109, 110, 111, 118, 119, 120, 121, 122, 123,
              130, 131, 132, 133, 134, 135, 142, 143, 144, 145, 146, 147, 154, 155, 156, 157, 158, 159]:
        df.drop(df_column_names[0][i - 1], axis=1, inplace=True)
    df.replace("?", np.nan, inplace=True)

    for column in df.columns:
        if df[column].dtype == "object":
            df[column] = df[column].astype("float")

        if df[column].dtype == "float64":
            df[column].fillna(df[column].mean(), inplace=True)

        if len(np.unique(df[column])) == 1:
            df.drop(column, axis=1, inplace=True)

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    significance_level = 0.05
    false_discovery_rate = SelectFdr(f_classif, alpha=significance_level)
    false_discovery_rate.fit(X, y)

    selected_features_mask = false_discovery_rate.get_support()
    selected_feature_indices = [i for (i, x) in enumerate(selected_features_mask) if x]

    df_new = df.iloc[:, selected_feature_indices]
    df_new.to_csv("/FilterWrapper/DatasetsFeatureSelection/Arrhythmia/arrhythmia_anova.csv", index=False)
