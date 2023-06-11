import random
from typing import Literal

import numpy as np
import pandas as pd


def split_input_target(df: pd.DataFrame, target_label: str) -> tuple[pd.DataFrame, np.ndarray]:
    target_index = df.columns.get_loc(key=target_label)

    X = df.iloc[:, [j for j in range(df.shape[1]) if j != target_index]]
    y = df.iloc[:, [target_index]].values.reshape(-1,).ravel()

    return X, y


def split_train_test_df_indices(df: pd.DataFrame, test_size=0.2) -> tuple[list[int], list[int]]:
    rows = df.shape[0]

    sample_testing_indices = random.sample(population=range(rows), k=int(test_size * rows))
    sample_training_indices = [i for i in range(rows) if i not in sample_testing_indices]

    return sample_training_indices, sample_testing_indices


def select_k_best_features_from_data_frame(df: pd.DataFrame, target_label: str, sorted_features: list[str],
                                           selected_feature_size=0.6) -> pd.DataFrame:
    k = int(selected_feature_size * len(sorted_features))
    selected_k_features_names = sorted_features[0:k]

    selected_k_features_indices = [
        i for i in range(df.shape[1])
        if df.columns[i] in selected_k_features_names or df.columns[i] == target_label]

    return df.iloc[:, selected_k_features_indices]


def drop_features(
        df: pd.DataFrame, target_label: str, feature_type: Literal["string", "int64", "float64"]) -> pd.DataFrame:
    target_index = df.columns.get_loc(key=target_label)
    actual_feature_type = "category" if feature_type == "string" else feature_type

    column_indices = [i for i, column in enumerate(
        df.columns) if df[column].dtype != actual_feature_type or i == target_index]

    return df.iloc[:, column_indices]


def drop_features_with_negative_values(df: pd.DataFrame, target_label: str) -> pd.DataFrame:
    def is_number(column_type) -> bool: return column_type == "int64" or column_type == "float64"
    def is_positive(column) -> bool: return all([x >= 0 for x in column])

    target_index = df.columns.get_loc(key=target_label)

    positive_column_indices = [i for i, column in enumerate(df.columns)
                               if is_number(df[column].dtype) and is_positive(df[column]) or i == target_index]

    return df.iloc[:, positive_column_indices]


def split_categorical_discrete_continuous_features(
        df: pd.DataFrame, target_label: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    target_index = df.columns.get_loc(key=target_label)

    categorical_column_indices = [i for i, column in enumerate(
        df.columns) if df[column].dtype == "category" or i == target_index]
    discrete_column_indices = [i for i, column in enumerate(
        df.columns) if df[column].dtype == "int64" or i == target_index]
    continuous_column_indices = [i for i, column in enumerate(
        df.columns) if df[column].dtype == "float64" or i == target_index]

    df_categorical = df.iloc[:, categorical_column_indices]
    df_discrete = df.iloc[:, discrete_column_indices]
    df_continuous = df.iloc[:, continuous_column_indices]

    return df_categorical, df_discrete, df_continuous
