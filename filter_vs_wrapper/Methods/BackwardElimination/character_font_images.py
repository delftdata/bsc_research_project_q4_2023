import pandas as pd
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.svm import LinearSVC
from sklearn.preprocessing import OrdinalEncoder
import random


def character_font_images_be():
    df_font_names = pd.read_csv("/FilterWrapper/Datasets/CharacterFontImages/font.names", header=None)
    frames: list[pd.DataFrame] = []

    for font_name in df_font_names[0]:
        df = pd.read_csv(f"/FilterWrapper/Datasets/CharacterFontImages/{font_name}")
        frames.append(df)

    df = pd.concat(frames)

    ordinal_encoder = OrdinalEncoder()

    for column in df.columns:
        if df[column].dtype == "object":
            df[column] = ordinal_encoder.fit_transform(df[column].values.reshape(-1, 1))

    df["font"] = df["font"].astype("int64").astype("category")

    row_sample_indices = random.sample(range(df.shape[0]), int(df.shape[0] * 0.0001))
    column_sample_indices = random.sample(range(1, df.shape[1]), int(df.shape[1] * 0.05))

    X = df.iloc[row_sample_indices, column_sample_indices]
    y = df.iloc[row_sample_indices, :1]

    import warnings
    warnings.filterwarnings("ignore")

    linear_svc = LinearSVC()
    backward_selector = SequentialFeatureSelector(
        linear_svc, scoring="accuracy", n_features_to_select=17, direction="backward")
    backward_selector.fit(X, y)

    selected_features_mask = backward_selector.get_support()
    selected_feature_indices = [0] + [column_sample_indices[i]
                                      for (i, x) in enumerate(selected_features_mask) if x]

    df_new = df.iloc[:, selected_feature_indices]
    df_new.to_csv("/FilterWrapper/DatasetsFeatureSelection/CharacterFontImages/character_font_images_be.csv",
                  index=False)
