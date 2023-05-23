from typing import Union

import pandas as pd
from autogluon.features.generators import (AutoMLPipelineFeatureGenerator,
                                           IdentityFeatureGenerator)
from autogluon.tabular import TabularDataset, TabularPredictor
from methods.filter import Filter
from methods.wrapper import Wrapper
from splitting.splitter import Splitter


class Evaluator:
    def __init__(self, df: pd.DataFrame, target_label: str, scoring: str, algorithm_names: list[tuple[str, str]]):
        self.df = df
        self.target_label = target_label
        self.scoring = scoring
        self.algorithm_names = algorithm_names

    def perform_experiments(self, method: Union[Filter, Wrapper]) -> dict[str, list[tuple[float, float]]]:
        percentage_range = [percentage / 100.0 for percentage in range(10, 110, 10)]
        performances: dict[str, list[tuple[float, float]]] = dict()

        for selected_feature_size in percentage_range:
            df = method.perform_feature_selection(self.df, selected_feature_size)
            print(df.head())

            try:
                results = self.evaluate_models(df)

                for (algorithm, performance) in results:
                    if algorithm not in performances.keys():
                        performances[algorithm] = []
                    performances[algorithm].append((float(performance[self.scoring]), selected_feature_size))

            except Exception as e:
                print(f"Autogluon: {e}")

        return performances

    def evaluate_models(self, df: pd.DataFrame, test_size=0.2) -> list[tuple[str, dict]]:
        results: list[tuple[str, dict]] = []
        sample_training_indices, sample_testing_indices = Splitter.split_train_test_df_indices(self.df, test_size)

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

        # svm_model = SVMModel(label=df.columns[0], problem_type="binary",
        #                      data_preprocessing=False)
        # svm_model.grid_search(df, param_grid=param_grid)

        # results.append(("SVM", performance))

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
