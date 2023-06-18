from __future__ import annotations

import pandas as pd
from autogluon.features.generators import IdentityFeatureGenerator
from autogluon.tabular import TabularDataset, TabularPredictor
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVC, SVR

from processing.preprocessing import discretize_columns_ordinal_encoder
from processing.splitter import (select_k_best_features_from_data_frame,
                                 split_input_target,
                                 split_train_test_df_indices)


class Evaluator:
    """A class that performs evaluation tasks on a given DataFrame for various algorithms.

    Attributes
    ----------
    df : pd.DataFrame
        The DataFrame containing the dataset.
    target_label : str
        The name of the target label column in the DataFrame.
    scoring : str
        The scoring metric to be used for evaluation.
    algorithm_names : list[tuple[str, str]]
        List of tuples containing the algorithms on which the performance of the feature selection techniques is evaluated.
        Each tuple has the following format: ("XGB", "XGBoost").
        The first element can be an arbitrary value, and the second element reflects the name of the algorithm used by
        Autogluon, thus it should be a valid algorithm name.
    """

    def __init__(
            self, df: pd.DataFrame, target_label: str, scoring: str, algorithm_names: list[tuple[str, str]],
            hyperparameters: dict):
        """
        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame containing the dataset.
        target_label : str
            The name of the target label column in the DataFrame.
        scoring : str
            The scoring metric to be used for evaluation.
        algorithm_names : list[tuple[str, str]]
            List of tuples containing the algorithms on which the performance of the feature selection techniques is evaluated.
            Each tuple has the following format: ("XGB", "XGBoost").
            The first element can be an arbitrary value, and the second element reflects the name of the algorithm used by
            Autogluon, thus it should be a valid algorithm name.

        """
        self.df = df
        self.target_label = target_label
        self.scoring = scoring
        self.algorithm_names = algorithm_names
        self.hyperparameters = hyperparameters

    def perform_experiments(self, sorted_features: list[str], svm=False) -> dict[str, list[float]]:
        """Performs experiments by selecting different feature sizes and evaluating models.

        For each selected feature size, the method selects the top features from the DataFrame based on the
        provided sorted feature list. It then evaluates the Autogluon models or SVM algorithm on the resulting
        DataFrame and collects the performance metrics.

        Parameters
        ----------
        sorted_features : list[str]
            A list of feature names sorted in descending order of importance.
        svm : bool, optional
            Flag indicating whether to perform experiments with the SVM algorithm (default: False).

        Returns
        -------
        dict[str, list[float]]
            A dictionary mapping algorithm names to lists of performance metric values.
            Each algorithm's performance is recorded for different feature sizes.
        """
        percentage_range = [percentage / 100.0 for percentage in range(10, 110, 10)]
        performance: dict[str, list[float]] = {}

        for selected_feature_size in percentage_range:
            df = select_k_best_features_from_data_frame(
                self.df, self.target_label, sorted_features, selected_feature_size)

            try:
                if svm:
                    results = self.evaluate_svm(df)
                else:
                    results = self.evaluate_models(df)

                for (algorithm, performance_algorithm) in results:
                    if algorithm not in performance:
                        performance[algorithm] = []
                    performance[algorithm].append(float(performance_algorithm[self.scoring]))

            except Exception as error:
                print(f"{error}")

        return performance

    def evaluate_svm(self, df: pd.DataFrame, test_size=0.2, max_rows=100, training_rounds=5) -> list[tuple[str, dict]]:
        """Evaluates an SVM model on the given dataset.

        Parameters
        ----------
        df : pd.DataFrame
            The dataset to evaluate the SVM model on.
        test_size : float, optional
            The proportion of the dataset to use for testing (default: 0.2).
        max_rows : int, optional
            The maximum number of rows to use for training when the dataset is larger than this value (default: 100).
        training_rounds : int, optional
            The number of training rounds to perform when the dataset size exceeds the `max_rows` value (default: 5).

        Returns
        -------
        list[tuple[str, dict]]
            A list of tuples containing the algorithm name ("SVM") and the performance dictionary for the SVM model.
        """
        results: list[tuple[str, dict]] = []
        df = discretize_columns_ordinal_encoder(df, [])
        sample_training_indices, sample_testing_indices = split_train_test_df_indices(self.df, test_size)
        train_data = TabularDataset(df.iloc[sample_training_indices, :])
        test_data = TabularDataset(df.iloc[sample_testing_indices, :])

        if self.scoring == "accuracy":
            predictor = SVC(C=self.hyperparameters["C"], gamma=self.hyperparameters["gamma"],
                            kernel=self.hyperparameters["kernel"])
        else:
            predictor = SVR(C=self.hyperparameters["C"], gamma=self.hyperparameters["gamma"],
                            kernel=self.hyperparameters["kernel"])

        if train_data.shape[0] > max_rows:
            actual_training_rounds = training_rounds
            train_data_sample = train_data.sample(n=max_rows, random_state=42)
        else:
            actual_training_rounds = 1
            train_data_sample = train_data

        for i in range(0, actual_training_rounds):
            print(f"{i + 1}/{actual_training_rounds}")
            X_train, y_train = split_input_target(train_data_sample, self.target_label)
            predictor.fit(X_train, y_train)
            if actual_training_rounds > 1:
                train_data_sample = train_data.sample(n=max_rows, random_state=42)

        X_test, y_test = split_input_target(test_data, self.target_label)
        y_pred = predictor.predict(X_test)

        performance_score = mean_squared_error(y_true=y_test, y_pred=y_pred)
        results.append(("SVM", {self.scoring: performance_score}))

        return results

    def evaluate_models(self, df: pd.DataFrame, test_size=0.2) -> list[tuple[str, dict]]:
        """Evaluates multiple models on the given dataset using the provided hyperparameters.

        Parameters
        ----------
        df : pd.DataFrame
            The dataset to evaluate the models on.
        test_size : float, optional
            The proportion of the dataset to use for testing (default: 0.2).

        Returns
        -------
        list[tuple[str, dict]]
            A list of tuples containing the algorithm name and the performance dictionary for each evaluated model.

        """
        results: list[tuple[str, dict]] = []
        sample_training_indices, sample_testing_indices = split_train_test_df_indices(self.df, test_size)
        train_data = TabularDataset(df.iloc[sample_training_indices, :])
        test_data = TabularDataset(df.iloc[sample_testing_indices, :])

        for (algorithm, _) in self.algorithm_names:
            predictor = TabularPredictor(label=self.target_label, eval_metric=self.scoring, verbosity=0)
            predictor.fit(train_data=train_data, presets="best_quality",
                          feature_generator=IdentityFeatureGenerator(),
                          hyperparameters={algorithm: self.hyperparameters[algorithm]})

            performance = predictor.evaluate(test_data)
            results.append((algorithm, performance))

        return results
