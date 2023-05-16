import random

import pandas as pd
from autogluon.features.generators import (AutoMLPipelineFeatureGenerator,
                                           IdentityFeatureGenerator)
from autogluon.tabular import TabularDataset, TabularPredictor
from methods.filter import Filter
from methods.wrapper import Wrapper
from sklearn.svm import LinearSVC, LinearSVR


class Evaluator:

    @staticmethod
    def perform_feature_selection_methods(
            df: pd.DataFrame, scoring: str, selected_features_size=0.6) -> list[tuple[str, pd.DataFrame]]:

        chi2, anova, forward_selection, backward_elimination = None, None, None, None

        try:
            chi2 = Filter.perform_feature_selection(
                df=df, filter_method="chi2", selected_features_size=selected_features_size)
        except Exception as e:
            print("Chi2:", e)

        try:
            anova = Filter.perform_feature_selection(
                df=df, filter_method="anova", selected_features_size=selected_features_size)
        except Exception as e:
            print("Anova:", e)

        try:
            forward_selection = Wrapper.perform_feature_selection(
                df=df, estimator=LinearSVC(), scoring=scoring, direction="forward",
                selected_features_size=selected_features_size)
        except Exception as e:
            print("Forward Selection:", e)

        try:
            backward_elimination = Wrapper.perform_feature_selection(
                df=df, estimator=LinearSVR(), scoring=scoring, direction="backward",
                selected_features_size=selected_features_size)
        except Exception as e:
            print("Backward Elimination:", e)

        data_frames: list[tuple[str, pd.DataFrame]] = [
            ("chi2", chi2),
            ("anova", anova),
            ("forward_selection", forward_selection),
            ("backward_elimination", backward_elimination),
            ("original", df)]

        selected_data_frames = [(method_name, df) for (method_name, df) in data_frames if df is not None]
        return selected_data_frames

    @staticmethod
    def evaluate_models(selected_data_frames: list[tuple[str, pd.DataFrame]],
                        algorithms_names: list[tuple[str, str]],
                        eval_metric: str,
                        test_size=0.2) -> list[tuple[str, str, str]]:

        original_df = selected_data_frames[len(selected_data_frames) - 1][1]
        target_label = original_df.columns[0]
        rows = original_df.shape[0]

        sample_training_indices = random.sample(population=range(rows), k=int((1 - test_size) * rows))
        sample_testing_indices = [i for i in range(rows) if i not in sample_training_indices]

        results: list[tuple[str, str, str]] = []

        for (method_name, df) in selected_data_frames:
            train_data_frame = df.iloc[sample_training_indices, :]
            test_data_frame = df.iloc[sample_testing_indices, :]

            for (algorithm, algorithm_name) in algorithms_names:
                hyperparameters = Evaluator.get_hyperparameters(
                    original_df, target_label, algorithm, algorithm_name, eval_metric)

                performance = Evaluator.evaluate_model(train_data_frame, test_data_frame, target_label,
                                                       algorithm, hyperparameters, eval_metric)

                results.append((method_name, algorithm, str(performance)))

            # svm_model = SVMModel(label=df.columns[0], problem_type="binary",
            #                      data_preprocessing=False)
            # svm_model.grid_search(df, param_grid=param_grid)

            # results.append((method_name, "SVM", str(performance)))

        return results

    @staticmethod
    def get_hyperparameters(
            df: pd.DataFrame, target_label: str, algorithm: str,
            algorithm_name: str, eval_metric: str) -> dict:

        auxiliary_data_frame = AutoMLPipelineFeatureGenerator(
            enable_text_special_features=False, enable_text_ngram_features=False)
        auxiliary_data_frame = auxiliary_data_frame.fit_transform(df)

        train_data = TabularDataset(auxiliary_data_frame)

        predictor = TabularPredictor(label=target_label, eval_metric=eval_metric, verbosity=0)
        predictor.fit(train_data=train_data, hyperparameters={algorithm: {}})

        training_results = predictor.info()
        return training_results["model_info"][algorithm_name]["hyperparameters"]

    @staticmethod
    def evaluate_model(
            train_data_frame: pd.DataFrame, test_data_frame: pd.DataFrame, target_label: str, algorithm: str,
            hyperparameters: dict, eval_metric: str) -> dict:

        train_data = TabularDataset(train_data_frame)
        test_data = TabularDataset(test_data_frame)

        predictor = TabularPredictor(label=target_label, eval_metric=eval_metric, verbosity=0)
        predictor.fit(train_data=train_data, presets="best_quality",
                      feature_generator=IdentityFeatureGenerator(), hyperparameters={algorithm: hyperparameters})

        return predictor.evaluate(test_data)
