import random

import numpy as np
import pandas as pd


class Splitter:

    @staticmethod
    def split_input_target(df: pd.DataFrame, target_label: str) -> tuple[pd.DataFrame, np.ndarray]:
        target_index = df.columns.get_loc(key=target_label)

        X = df.iloc[:, [j for j in range(df.shape[1]) if j != target_index]]
        y = df.iloc[:, [target_index]].values.reshape(-1,).ravel()

        return X, y

    @staticmethod
    def split_train_test_df_indices(df: pd.DataFrame, test_size=0.2) -> tuple[list[int], list[int]]:
        rows = df.shape[0]

        sample_testing_indices = random.sample(population=range(rows), k=int(test_size * rows))
        sample_training_indices = [i for i in range(rows) if i not in sample_testing_indices]

        return sample_training_indices, sample_testing_indices
