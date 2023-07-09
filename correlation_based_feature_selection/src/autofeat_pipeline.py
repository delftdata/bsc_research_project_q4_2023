import pandas as pd
import numpy as np
import os
import time
import csv
from sklearn.model_selection import train_test_split
from autogluon.features.generators import AutoMLPipelineFeatureGenerator, FillNaFeatureGenerator
from autogluon.tabular import TabularDataset
from autogluon.tabular import TabularPredictor
from .correlation_methods.pearson import PearsonFeatureSelection
from .correlation_methods.spearman import SpearmanFeatureSelection
from .correlation_methods.su import SymmetricUncertaintyFeatureSelection
from .correlation_methods.information_gain import InformationGainFeatureSelection
from .correlation_methods.relief import ReliefFeatureSelection
from warnings import filterwarnings
from sklearn.datasets import fetch_openml


filterwarnings("ignore", category=UserWarning)
filterwarnings("ignore", category=RuntimeWarning)
filterwarnings("ignore", category=FutureWarning)


class PreML:
    @staticmethod
    def imputation_most_common_value(dataframe):
        return dataframe.apply(lambda x: x.fillna(x.value_counts().index[0]))


class InML:
    @staticmethod
    def feature_selection_select_k_best(train_dataframe, target_label):
        start = time.time()
        pearson_ranked_features, pearson_correlations = \
            PearsonFeatureSelection.feature_selection(train_dataframe, target_label)
        pearson_duration = time.time() - start

        start = time.time()
        spearman_ranked_features, spearman_correlations = \
            SpearmanFeatureSelection.feature_selection(train_dataframe, target_label)
        spearman_duration = time.time() - start

        start = time.time()
        su_ranked_features, su_correlations = \
            SymmetricUncertaintyFeatureSelection.feature_selection(train_dataframe, target_label)
        su_duration = time.time() - start

        start = time.time()
        ig_ranked_features, ig_correlations = \
            InformationGainFeatureSelection.feature_selection(train_dataframe, target_label)
        ig_duration = time.time() - start

        return pearson_ranked_features, spearman_ranked_features, su_ranked_features, ig_ranked_features, \
            pearson_correlations, spearman_correlations, su_correlations, ig_correlations, \
            pearson_duration, spearman_duration, su_duration, ig_duration

    @staticmethod
    def feature_selection_select_k_best_relief(dataset_name, train_dataframe, target_label, number_of_features_k):
        # Establish the problem type
        problem_type = 'binary_classification'
        if dataset_name == 'HousingPrices' or dataset_name == 'TOPO-2-1' or dataset_name == 'QSAR-TID-11109':
            problem_type = 'regression'

        start = time.time()
        relief_ranked_features, relief_correlations = \
            ReliefFeatureSelection.feature_selection(train_dataframe, target_label, number_of_features_k, problem_type)
        relief_duration = time.time() - start

        return relief_ranked_features, relief_correlations, relief_duration


class PostML:
    @staticmethod
    def evaluate_model(algorithm, hyperparameters, train_dataframe, feature_subset,
                       target_label, test_dataframe, evaluation_metric):
        train_data = TabularDataset(train_dataframe[feature_subset])
        start_time_model = time.time()
        fitted_predictor = TabularPredictor(label=target_label,
                                            eval_metric=evaluation_metric,
                                            verbosity=1) \
            .fit(train_data=train_data, hyperparameters={algorithm: hyperparameters})

        # Get the duration the model with feature selection took
        current_duration = time.time() - start_time_model

        # Evaluate the model with feature selection applied
        test_data = TabularDataset(test_dataframe)
        current_performance = fitted_predictor.evaluate(test_data)[evaluation_metric]
        current_performance = abs(current_performance)

        return current_performance, current_duration


class MLPipeline:
    def __init__(self, dataset_file, dataset_name, target_label, evaluation_metric, features_to_select='small'):
        self.dataset_file = dataset_file
        self.dataset_name = dataset_name
        self.target_label = target_label
        self.dataframe = None
        self.auxiliary_dataframe = None
        if dataset_name != 'QSAR-TID-11109':
            self.dataframe = pd.read_csv(dataset_file)
            self.auxiliary_dataframe = pd.read_csv(dataset_file)
        else:
            qsar = fetch_openml("QSAR-TID-11109", as_frame=False)
            data = qsar.data
            target = qsar.target
            feature_names = qsar.feature_names
            target_name = qsar.target_names
            dense_data = data.toarray()
            dataframe = pd.DataFrame(dense_data, columns=feature_names)
            dataframe[target_name[0]] = target
            self.dataframe = dataframe
            self.auxiliary_dataframe = dataframe

        # Specify the models to use: GBM (LightGBM), RF (RandomForest), LR (LinearModel), XGB (XGBoost)
        self.algorithms_model_names = {
            'GBM': 'LightGBM',
            # 'RF': 'RandomForest',
            # 'LR': 'LinearModel',
            # 'XGB': 'XGBoost'
        }
        self.evaluation_metric = evaluation_metric

        # The number of features that will be selected during feature selection (excl. target)
        self.features_to_select_k = [self.dataframe.shape[1] - 1]
        # if features_to_select == 'small':
        #     number_columns = self.dataframe.shape[1] - 1
        #     self.features_to_select_k = [5]
        #     self.features_to_select_k += list(range(10, number_columns, 10))
        #     # self.features_to_select_k += [number_columns]
        # elif features_to_select == 'medium':
        #     number_columns = self.dataframe.shape[1] - 1
        #     self.features_to_select_k = [5, 10, 25]
        #     self.features_to_select_k += list(range(50, number_columns, 50))
        #     # self.features_to_select_k += [number_columns]
        # elif features_to_select == 'large':
        #     number_columns = self.dataframe.shape[1] - 1
        #     self.features_to_select_k = [5, 10, 25, 50, 100, 250, 500]
        #     self.features_to_select_k += list(range(1000, number_columns, 1000))
        #     # self.features_to_select_k += [number_columns]

    def run_model_no_feature_selection(self, algorithm, model_name, train_dataframe, test_dataframe):
        train_dataframe = TabularDataset(train_dataframe)
        start_time_baseline = time.time()
        fitted_predictor = TabularPredictor(label=self.target_label,
                                            eval_metric=self.evaluation_metric,
                                            verbosity=1) \
            .fit(train_data=train_dataframe, hyperparameters={algorithm: {}}, feature_generator=None)
        # Get the duration the baseline took
        baseline_duration = time.time() - start_time_baseline

        # Get the tuned hyperparameters
        training_results = fitted_predictor.info()
        hyperparameters = training_results['model_info'][model_name]['hyperparameters']

        # Get the performance and the feature importance given by the baseline model
        importance = fitted_predictor.feature_importance(data=test_dataframe, feature_stage='original')
        baseline_performance = fitted_predictor.evaluate(test_dataframe)[self.evaluation_metric]
        baseline_performance = abs(baseline_performance)

        print("Feature importance: ")
        print(importance)

        return hyperparameters, baseline_performance, baseline_duration

    def evaluate_all_models(self):
        # Print information about the dataset
        number_rows, number_columns = self.dataframe.shape
        print('Dataset: ' + self.dataset_name)
        print('Total columns: ' + str(number_columns - 1))
        print('Total rows: ' + str(number_rows))
        print('Feature subsets: ' + str(self.features_to_select_k))

        # Prepare the data for Autogluon
        self.dataframe = TabularDataset(self.dataframe)
        self.dataframe = FillNaFeatureGenerator(inplace=True).fit_transform(self.dataframe)
        self.auxiliary_dataframe = AutoMLPipelineFeatureGenerator(
            enable_text_special_features=False,
            enable_text_ngram_features=False) \
            .fit_transform(self.dataframe)
        self.auxiliary_dataframe = PreML.imputation_most_common_value(self.auxiliary_dataframe)

        # Split the data into train and test
        x_train, x_test, y_train, y_test = \
            train_test_split(self.auxiliary_dataframe.drop(columns=[self.target_label]),
                             self.auxiliary_dataframe[self.target_label],
                             test_size=0.2, random_state=42)
        current_train_dataframe = pd.concat([x_train, y_train], axis=1)
        current_test_dataframe = pd.concat([x_test, y_test], axis=1)

        # COMPUTATION: Compute the ranking of features returned by each correlation method
        pearson_selected_features, spearman_selected_features, su_selected_features, ig_selected_features, \
            pearson_correlations, spearman_correlations, su_correlations, ig_correlations, \
            pearson_duration, spearman_duration, su_duration, ig_duration = \
            InML.feature_selection_select_k_best(current_train_dataframe,
                                                 self.target_label)

        # LOOP: Go through each algorithm
        for algorithm, algorithm_name in self.algorithms_model_names.items():
            # COMPUTATION: Get the hyperparameters on the dataset with all features (i.e. baseline configuration)
            hyperparameters, baseline_performance, baseline_duration = \
                self.run_model_no_feature_selection(algorithm, algorithm_name,
                                                    current_train_dataframe, current_test_dataframe)

            # LOOP: Go through each method
            correlation_methods = ['Pearson', 'Spearman', 'SU', 'IG']
            for ranked_features, correlation_values, correlation_method in \
                    zip([pearson_selected_features, spearman_selected_features,
                         su_selected_features, ig_selected_features],
                        [pearson_correlations, spearman_correlations,
                         su_correlations, ig_correlations],
                        correlation_methods):
                # LOOP: Go to all possible values of k (i.e. number of selected features)
                # k depends on whether the dataset is small, medium or large
                for subset_length in self.features_to_select_k:
                    # Get the current feature subset
                    current_subset = ranked_features #[:subset_length]
                    current_correlation_values = correlation_values #[:subset_length]
                    paired_values = list(zip(current_subset, current_correlation_values))
                    current_subset.append(self.target_label)

                    current_performance, current_duration = PostML.evaluate_model(algorithm=algorithm,
                                                                                  hyperparameters=hyperparameters,
                                                                                  train_dataframe=
                                                                                  current_train_dataframe,
                                                                                  feature_subset=current_subset,
                                                                                  target_label=self.target_label,
                                                                                  test_dataframe=
                                                                                  current_test_dataframe,
                                                                                  evaluation_metric=
                                                                                  self.evaluation_metric)

                    total_duration = current_duration
                    if correlation_method == 'Pearson':
                        total_duration += pearson_duration
                    elif correlation_method == 'Spearman':
                        total_duration += spearman_duration
                    elif correlation_method == 'SU':
                        total_duration += su_duration
                    elif correlation_method == 'IG':
                        total_duration += ig_duration
                    # Save the results to file
                    MLPipeline.write_to_file(dataset_name=self.dataset_name,
                                             algorithm_name=algorithm_name,
                                             correlation_method=correlation_method,
                                             subset_length=subset_length,
                                             current_subset=current_subset,
                                             current_correlations=paired_values,
                                             current_performance=current_performance,
                                             current_duration=total_duration,
                                             baseline_performance=baseline_performance,
                                             baseline_duration=baseline_duration)

            for subset_length in self.features_to_select_k:
                relief_ranked_features, relief_correlations, relief_duration = \
                    InML.feature_selection_select_k_best_relief(dataset_name=self.dataset_name,
                                                                train_dataframe=current_train_dataframe,
                                                                target_label=self.target_label,
                                                                number_of_features_k=subset_length)

                current_subset = relief_ranked_features[:subset_length]
                current_subset.append(self.target_label)
                current_correlation_values = relief_correlations[:subset_length]
                paired_values = list(zip(current_subset, current_correlation_values))

                current_performance, current_duration = PostML.evaluate_model(algorithm=algorithm,
                                                                              hyperparameters=hyperparameters,
                                                                              train_dataframe=
                                                                              current_train_dataframe,
                                                                              feature_subset=current_subset,
                                                                              target_label=self.target_label,
                                                                              test_dataframe=
                                                                              current_test_dataframe,
                                                                              evaluation_metric=
                                                                              self.evaluation_metric)
                total_duration = current_duration + relief_duration
                # Save the results to file
                MLPipeline.write_to_file(dataset_name=self.dataset_name,
                                         algorithm_name=algorithm_name,
                                         correlation_method='Relief',
                                         subset_length=subset_length,
                                         current_subset=current_subset,
                                         current_correlations=paired_values,
                                         current_performance=current_performance,
                                         current_duration=total_duration,
                                         baseline_performance=baseline_performance,
                                         baseline_duration=baseline_duration)

    @staticmethod
    def write_to_file(dataset_name, algorithm_name, correlation_method,
                      subset_length, current_subset, current_correlations,
                      current_performance, current_duration, baseline_performance, baseline_duration):
        # Create the directory if it doesn't exist
        directory = "./autofeat_results"
        os.makedirs(directory, exist_ok=True)

        # Write the results to a txt file
        file_path = f"./autofeat_results/{dataset_name}_{algorithm_name}_" \
                    f"{correlation_method}.txt"
        file = open(file_path, "a")

        file.write("DATASET NAME: " + dataset_name + '\n')
        file.write("ALGORITHM NAME: " + algorithm_name + '\n')
        file.write("CORRELATION METHOD: " + correlation_method + '\n')
        file.write("SUBSET OF FEATURES: " + str(subset_length) + '\n')
        file.write("CURRENT FEATURE SUBSET: " + str(current_subset) + '\n')
        file.write("CURRENT FEATURE SUBSET CORRELATIONS: " + str(current_correlations) + '\n')
        file.write("CURRENT PERFORMANCE: " + str(current_performance) + '\n')
        file.write("CURRENT RUNTIME: " + str(current_duration) + '\n')
        file.write("BASELINE PERFORMANCE: " + str(baseline_performance) + '\n')
        file.write("BASELINE RUNTIME: " + str(baseline_duration) + '\n')
        file.write('\n')
        file.close()

        # Write all results to a csv file
        # csv_file_path = f"./autofeat_results/all_results.csv"
        # csv_file_exists = os.path.exists(csv_file_path)
        #
        # with open(csv_file_path, "a", newline='') as csv_file_path:
        #     writer = csv.writer(csv_file_path)
        #     if not csv_file_exists:
        #         writer.writerow(["DATASET NAME", "ALGORITHM NAME", "CORRELATION METHOD",
        #                          "SUBSET OF FEATURES", "CURRENT FEATURE SUBSET",
        #                          "CURRENT FEATURE SUBSET CORRELATIONS",
        #                          "CURRENT PERFORMANCE", "CURRENT RUNTIME",
        #                          "BASELINE PERFORMANCE", "BASELINE RUNTIME"])
        #     writer.writerow([dataset_name, algorithm_name, correlation_method,
        #                      subset_length, current_subset, current_correlations,
        #                      current_performance, current_duration,
        #                      baseline_performance, baseline_duration])
