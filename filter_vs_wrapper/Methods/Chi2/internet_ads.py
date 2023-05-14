import numpy as np
import pandas as pd
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectFdr


def internet_ads_chi2():
    df = pd.read_csv("/FilterWrapper/Datasets/InternetAdvertisements/ad.data",
                     header=None, low_memory=False)
    df_column_names = pd.read_csv("/FilterWrapper/Datasets/InternetAdvertisements/ad.names", header=None)

    df.rename(columns=dict([(column, df_column_names[0][column]) for column in range(df.shape[1])]), inplace=True)
    df["ad"].replace("ad.", 1, inplace=True)
    df["ad"].replace("nonad.", 0, inplace=True)
    df.replace("?", np.nan, inplace=True)
    df.replace("   ?", np.nan, inplace=True)
    df.dropna(inplace=True)

    for column in df.columns:
        if df[column].dtype == "object":
            df[column] = df[column].astype("float")

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    significance_level = 0.05
    false_discovery_rate = SelectFdr(chi2, alpha=significance_level)
    false_discovery_rate.fit(X, y)

    selected_features_mask = false_discovery_rate.get_support()
    selected_feature_indices = [-1] + [i for (i, x) in enumerate(selected_features_mask) if x]

    df_new = df.iloc[:, selected_feature_indices]
    df_new.to_csv("/FilterWrapper/DatasetsFeatureSelection/InternetAdvertisements/internet_ads_chi2.csv", index=False)
