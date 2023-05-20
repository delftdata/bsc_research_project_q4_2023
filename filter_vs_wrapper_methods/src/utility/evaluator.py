import random
from typing import Literal

import pandas as pd
from autogluon.features.generators import (AutoMLPipelineFeatureGenerator,
                                           IdentityFeatureGenerator)
from autogluon.tabular import TabularDataset, TabularPredictor
from methods.filter import Filter
from methods.wrapper import Wrapper


class Evaluator:
    def __init__(self, df: pd.DataFrame, target_label: str,
                 scoring: Literal["accuracy", "neg_mean_squared_error"],
                 algorithms_names: list[tuple[str, str]] = [
                     ("GBM", "LightGBM"), ("RF", "RandomForest"), ("LR", "LinearModel"), ("XGB", "XGBoost")],
                 filter_methods: list[Literal["Chi2", "ANOVA"]] = ["Chi2", "ANOVA"],
                 wrapper_methods:
                 list[Literal["Forward Selection", "Backward Elimination"]] = ["Forward Selection",
                                                                               "Backward Elimination"]):
        self.df = df
        self.target_label = target_label
        self.scoring: Literal['accuracy', 'neg_mean_squared_error'] = scoring
        self.algorithms_names = algorithms_names

        self.filter_methods: list[Filter] = []
        self.wrapper_methods: list[Wrapper] = []

        for filter_method in filter_methods:
            try:
                filter_method_instance = Filter(df=self.df, method=filter_method, target_label=self.target_label)
                self.filter_methods.append(filter_method_instance)
            except Exception as e:
                print(f"{filter_method}: {e}")

        for wrapper_method in wrapper_methods:
            try:
                wrapper_method_instance = Wrapper(df=self.df, method=wrapper_method,
                                                  target_label=self.target_label, scoring=self.scoring)
                self.wrapper_methods.append(wrapper_method_instance)
            except Exception as e:
                print(f"{wrapper_method}: {e}")

    def perform_experiments(self, percentage_key="Percentage of selected features"
                            ) -> dict[str, dict[str, list[float]]]:

        performances: dict[str, dict[str, list[float]]] = dict()
        percentage_range = [percentage / 100.0 for percentage in range(10, 110, 10)]

        for selected_feature_size in percentage_range:
            selected_filter_data_frames = self.perform_filter_feature_selection_methods(
                selected_features_size=selected_feature_size)
            selected_wrapper_data_frames = self.perform_wrapper_feature_selection_methods(
                selected_features_size=selected_feature_size)
            selected_data_frames = selected_filter_data_frames + selected_wrapper_data_frames

            for (method_name, df) in selected_data_frames:
                print(method_name)
                print(df.head())

            try:
                results = self.evaluate_models(selected_data_frames=selected_data_frames)

                for (method_name, algorithm, performance) in results:
                    if algorithm not in performances.keys():
                        performances[algorithm] = dict()
                    if method_name not in performances[algorithm].keys():
                        performances[algorithm][method_name] = []
                    if percentage_key not in performances[algorithm].keys():
                        performances[algorithm][percentage_key] = []
                    performances[algorithm][method_name].append(float(performance[self.scoring]))
                    if selected_feature_size not in performances[algorithm][percentage_key]:
                        performances[algorithm][percentage_key].append(selected_feature_size)

            except Exception as e:
                print(f"Autogluon: {e}")

        return performances

    def perform_filter_feature_selection_methods(self, selected_features_size: float) -> list[tuple[str, pd.DataFrame]]:
        return [(filter_method.method,
                 filter_method.perform_feature_selection(selected_features_size=selected_features_size))
                for filter_method in self.filter_methods]

    def perform_wrapper_feature_selection_methods(self, selected_features_size: float) -> list[tuple
                                                                                               [str, pd.DataFrame]]:
        return [(wrapper_method.method,
                 wrapper_method.perform_feature_selection(selected_features_size=selected_features_size))
                for wrapper_method in self.wrapper_methods]

    def evaluate_models(self, selected_data_frames: list[tuple[str, pd.DataFrame]],
                        test_size=0.2) -> list[tuple[str, str, dict]]:

        rows = self.df.shape[0]
        sample_training_indices = random.sample(population=range(rows), k=int((1 - test_size) * rows))
        sample_testing_indices = [i for i in range(rows) if i not in sample_training_indices]

        results: list[tuple[str, str, dict]] = []

        for (method_name, selected_df) in selected_data_frames:
            train_data = TabularDataset(selected_df.iloc[sample_training_indices, :])
            test_data = TabularDataset(selected_df.iloc[sample_testing_indices, :])

            for (algorithm, algorithm_name) in self.algorithms_names:
                hyperparameters = self.get_hyperparameters(
                    df=self.df, algorithm=algorithm, algorithm_name=algorithm_name)

                predictor = TabularPredictor(label=self.target_label, eval_metric=self.scoring, verbosity=0)
                predictor.fit(train_data=train_data, presets="best_quality",
                              feature_generator=IdentityFeatureGenerator(),
                              hyperparameters={algorithm: hyperparameters})

                performance = predictor.evaluate(test_data)
                results.append((method_name, algorithm, performance))

            # svm_model = SVMModel(label=df.columns[0], problem_type="binary",
            #                      data_preprocessing=False)
            # svm_model.grid_search(df, param_grid=param_grid)

            # results.append((method_name, "SVM", performance))

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
