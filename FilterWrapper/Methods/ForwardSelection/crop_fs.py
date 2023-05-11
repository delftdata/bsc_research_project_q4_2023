import pandas as pd
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.svm import LinearSVC
import random


def crop_fs():
    frames: list[pd.DataFrame] = []

    for i in range(10):
        df = pd.read_csv(f"/FilterWrapper/Datasets/Crop/crop{i}.csv")
        frames.append(df)

    df = pd.concat(frames)

    for column in df.columns[1:]:
        if df[column].dtype != "float64":
            df[column] = df[column].astype("float")

    df["label"] = df["label"].astype("category")

    row_sample_indices = random.sample(range(df.shape[0]), int(df.shape[0] * 0.0002))

    X = df.iloc[row_sample_indices, 1:]
    y = df.iloc[row_sample_indices, :1]

    import warnings
    warnings.filterwarnings("ignore")

    linear_svc = LinearSVC()
    forward_selector = SequentialFeatureSelector(
        linear_svc, scoring="accuracy", n_features_to_select=3)
    forward_selector.fit(X, y)

    selected_features_mask = forward_selector.get_support()
    selected_feature_indices = [0] + [i for (i, x) in enumerate(selected_features_mask) if x]

    df_new = df.iloc[:, selected_feature_indices]
    df_new.to_csv("/FilterWrapper/DatasetsFeatureSelection/Crop/crop.csv", index=False)
