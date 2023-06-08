from __future__ import annotations

import pandas as pd
from autogluon.features.generators import (AutoMLPipelineFeatureGenerator,
                                           IdentityFeatureGenerator)
from autogluon.tabular import TabularDataset, TabularPredictor
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC, SVR

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
    svm_param_grid : list[dict[str, list[str | int | float]]]
        svm_param_grid : list[dict[str, list[str | int | float]]]
        List of dictionaries where the keys are SVM hyperparameter names, and the values are possible values that these
        hyperparameters can take.
    """

    def __init__(
            self, df: pd.DataFrame, target_label: str, scoring: str, algorithm_names: list[tuple[str, str]],
            svm_param_grid: list[dict[str, list[str | int | float]]]):
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
        svm_param_grid : list[dict[str, list[str | int | float]]]
            svm_param_grid : list[dict[str, list[str | int | float]]]
            List of dictionaries where the keys are SVM hyperparameter names, and the values are possible values that these
            hyperparameters can take.
        """
        self.df = df
        self.target_label = target_label
        self.scoring = scoring
        self.algorithm_names = algorithm_names
        self.svm_param_grid = svm_param_grid

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
        performance: dict[str, list[float]] = dict()

        for selected_feature_size in percentage_range:
            df = select_k_best_features_from_data_frame(
                self.df, self.target_label, sorted_features, selected_feature_size)

            try:
                results = self.evaluate_models(df) if not svm else self.evaluate_svm(df)

                for (algorithm, performance_algorithm) in results:
                    if algorithm not in performance:
                        performance[algorithm] = []
                    performance[algorithm].append(float(performance_algorithm[self.scoring]))

            except Exception as e:
                print(f"Autogluon: {e}")

        return performance

    def evaluate_svm(self, df: pd.DataFrame, test_size=0.2) -> list[tuple[str, dict]]:
        """Evaluates the SVM algorithm on the given DataFrame.

        The method splits the DataFrame into training and testing sets, performs hyperparameter tuning using
        GridSearchCV, and evaluates the SVM algorithm on the testing set. The performance score is computed
        based on the specified scoring metric during the initialization of the Evaluator object.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame containing the dataset.
        test_size : float, optional
            The proportion of the dataset to include in the testing set (default: 0.2).

        Returns
        -------
        list[tuple[str, dict]]
            A list of tuples containing the algorithm name and a dictionary of performance metrics.
            The performance metrics include the specified scoring metric and its corresponding value.
        """
        results: list[tuple[str, dict]] = []
        sample_training_indices, sample_testing_indices = split_train_test_df_indices(self.df, test_size)
        train_data = TabularDataset(df.iloc[sample_training_indices, :])
        test_data = TabularDataset(df.iloc[sample_testing_indices, :])

        estimator = SVC() if self.scoring == "accuracy" else SVR()
        grid = GridSearchCV(estimator, self.svm_param_grid, refit=True, cv=5, n_jobs=-1)
        X_train, y_train = split_input_target(train_data, self.target_label)
        grid.fit(X_train, y_train)
        svm_hyperparameters = grid.best_params_

        if self.scoring == "accuracy":
            predictor = SVC(C=svm_hyperparameters["C"], gamma=svm_hyperparameters["gamma"],
                            kernel=svm_hyperparameters["kernel"])
        else:
            predictor = SVR(C=svm_hyperparameters["C"], gamma=svm_hyperparameters["gamma"],
                            kernel=svm_hyperparameters["kernel"])

        predictor.fit(X_train, y_train)
        X_test, y_test = split_input_target(test_data, self.target_label)
        performance_score = predictor.score(X_test, y_test)
        results.append(("SVM", {self.scoring: performance_score}))

        return results

    def evaluate_models(self, df: pd.DataFrame, test_size=0.2) -> list[tuple[str, dict]]:
        """Evaluates multiple models on the given DataFrame.

        The method splits the DataFrame into training and testing sets, trains and evaluates each model
        specified in the `algorithm_names` attribute using AutoGluon. The performance metrics are computed
        based on the specified scoring metric during the initialization of the Evaluator object.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame containing the dataset.
        test_size : float, optional
            The proportion of the dataset to include in the testing set (default: 0.2).

        Returns
        -------
        list[tuple[str, dict]]
            A list of tuples containing the algorithm name and a dictionary of performance metrics.
            The performance metrics include the specified scoring metric and its corresponding value.
        """
        results: list[tuple[str, dict]] = []
        sample_training_indices, sample_testing_indices = split_train_test_df_indices(self.df, test_size)
        train_data = TabularDataset(df.iloc[sample_training_indices, :])
        test_data = TabularDataset(df.iloc[sample_testing_indices, :])

        for (algorithm, algorithm_name) in self.algorithm_names:
            hyperparameters = self.get_hyperparameters(
                df=self.df, algorithm=algorithm, algorithm_name=algorithm_name)

            predictor = TabularPredictor(label=self.target_label, eval_metric=self.scoring, verbosity=0)
            predictor.fit(train_data=train_data, presets="best_quality",
                          feature_generator=IdentityFeatureGenerator(),
                          hyperparameters={algorithm: hyperparameters})

            performance = predictor.evaluate(test_data)
            results.append((algorithm, performance))

        return results

    def get_hyperparameters(self, df: pd.DataFrame, algorithm: str, algorithm_name: str) -> dict:
        """Picks the hyperparameters for the specified algorithm.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame containing the dataset.
        algorithm : str
            The abbreviation of the algorithm. This can be an arbitrary value.
        algorithm_name : str
            The algorithm key corresponding to the AutoGluon algorithm.

        Returns
        -------
        dict
            A dictionary of hyperparameters for the specified algorithm.
        """
        auxiliary_data_frame = AutoMLPipelineFeatureGenerator(
            enable_text_special_features=False, enable_text_ngram_features=False)
        auxiliary_data_frame = auxiliary_data_frame.fit_transform(df)

        train_data = TabularDataset(auxiliary_data_frame)

        predictor = TabularPredictor(label=self.target_label, eval_metric=self.scoring, verbosity=0)
        predictor.fit(train_data=train_data, hyperparameters={algorithm: {}})

        training_results = predictor.info()
        return training_results["model_info"][algorithm_name]["hyperparameters"]
