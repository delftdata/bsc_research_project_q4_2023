import pandas as pd
from autogluon.features.generators import (AutoMLPipelineFeatureGenerator,
                                           IdentityFeatureGenerator)
from autogluon.tabular import TabularDataset, TabularPredictor
from processing.splitter import (select_k_best_features_from_data_frame,
                                 split_train_test_df_indices)


class Evaluator:
    def __init__(self, df: pd.DataFrame, target_label: str, scoring: str, algorithm_names: list[tuple[str, str]]):
        self.df = df
        self.target_label = target_label
        self.scoring = scoring
        self.algorithm_names = algorithm_names

    def perform_experiments(self, sorted_features: list[str]) -> dict[str, list[float]]:
        percentage_range = [percentage / 100.0 for percentage in range(10, 110, 10)]
        performance: dict[str, list[float]] = dict()

        for selected_feature_size in percentage_range:
            df = select_k_best_features_from_data_frame(
                self.df, self.target_label, sorted_features, selected_feature_size)
            # print(df.head())

            try:
                results = self.evaluate_models(df)

                for (algorithm, performance_algorithm) in results:
                    if algorithm not in performance.keys():
                        performance[algorithm] = []
                    performance[algorithm].append(float(performance_algorithm[self.scoring]))

            except Exception as e:
                print(f"Autogluon: {e}")

        return performance

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
