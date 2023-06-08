from typing import Union

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
    def __init__(
            self, df: pd.DataFrame, target_label: str, scoring: str, algorithm_names: list[tuple[str, str]],
            svm_param_grid: list[dict[str, list[Union[str, int, float]]]]):
        self.df = df
        self.target_label = target_label
        self.scoring = scoring
        self.algorithm_names = algorithm_names
        self.svm_param_grid = svm_param_grid

    def perform_experiments(self, sorted_features: list[str], svm=False) -> dict[str, list[float]]:
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
        auxiliary_data_frame = AutoMLPipelineFeatureGenerator(
            enable_text_special_features=False, enable_text_ngram_features=False)
        auxiliary_data_frame = auxiliary_data_frame.fit_transform(df)

        train_data = TabularDataset(auxiliary_data_frame)

        predictor = TabularPredictor(label=self.target_label, eval_metric=self.scoring, verbosity=0)
        predictor.fit(train_data=train_data, hyperparameters={algorithm: {}})

        training_results = predictor.info()
        return training_results["model_info"][algorithm_name]["hyperparameters"]
