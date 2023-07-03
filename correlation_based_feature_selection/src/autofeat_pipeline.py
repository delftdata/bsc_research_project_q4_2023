import pandas as pd
import numpy as np
import os
import time
import timeit
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from autogluon.features.generators import AutoMLPipelineFeatureGenerator, FillNaFeatureGenerator
from autogluon.tabular import TabularDataset
from sklearn.preprocessing import MinMaxScaler
from autogluon.tabular import TabularPredictor
from .correlation_methods.pearson import PearsonFeatureSelection
from .correlation_methods.spearman import SpearmanFeatureSelection
from .correlation_methods.su import SymmetricUncertaintyFeatureSelection
from .correlation_methods.information_gain import InformationGainFeatureSelection
from warnings import filterwarnings
from sklearn.svm import LinearSVR, LinearSVC


filterwarnings("ignore", category=UserWarning)
filterwarnings("ignore", category=RuntimeWarning)
filterwarnings("ignore", category=FutureWarning)


class PreML:
    @staticmethod
    def imputation_most_common_value(dataframe):
        return dataframe.apply(lambda x: x.fillna(x.value_counts().index[0]))


class InML:
    @staticmethod
    def feature_selection_select_k_best(train_dataframe, target_label, k):
        start = time.time()
        pearson_selected_features = PearsonFeatureSelection.feature_selection(train_dataframe, target_label, k)
        pearson_duration = time.time() - start

        start = time.time()
        spearman_selected_features = SpearmanFeatureSelection.feature_selection(train_dataframe, target_label, k)
        spearman_duration = time.time() - start

        start = time.time()
        su_selected_features = SymmetricUncertaintyFeatureSelection.feature_selection(train_dataframe, target_label, k)
        su_duration = time.time() - start

        start = time.time()
        ig_selected_features = InformationGainFeatureSelection.feature_selection(train_dataframe, target_label, k)
        ig_duration = time.time() - start

        return pearson_selected_features, spearman_selected_features, su_selected_features, ig_selected_features, \
            pearson_duration, spearman_duration, su_duration, ig_duration


class PostML:
    @staticmethod
    def evaluate_model(algorithm, hyperparameters, train_dataframe, feature_subset,
                       target_label, test_dataframe, evaluation_metric):
        train_data = TabularDataset(train_dataframe[feature_subset])
        start_time_model = time.time()
        fitted_predictor = TabularPredictor(label=target_label,
                                            eval_metric=evaluation_metric,
                                            verbosity=0) \
            .fit(train_data=train_data, hyperparameters={algorithm: hyperparameters})

        # Get the duration the model with feature selection took
        current_duration = time.time() - start_time_model

        # Evaluate the model with feature selection applied
        test_data = TabularDataset(test_dataframe)
        current_performance = fitted_predictor.evaluate(test_data)[evaluation_metric]

        return current_performance, current_duration


class MLPipeline:
    def __init__(self, dataset_file, dataset_name, target_label, evaluation_metric, features_to_select=None):
        self.dataset_file = dataset_file
        self.dataset_name = dataset_name
        self.target_label = target_label
        self.dataframe = pd.read_csv(dataset_file)
        self.auxiliary_dataframe = pd.read_csv(dataset_file)
        # Specify the models to use: GBM (LightGBM), RF (RandomForest), LR (LinearModel), XGB (XGBoost)
        self.algorithms_model_names = {
            'GBM': 'LightGBM',
            # 'RF': 'RandomForest',
            # 'LR': 'LinearModel',
            # 'XGB': 'XGBoost'
        }
        # The maximum number of features that can be selected during feature selection (excl. target)
        self.features_to_select_k = features_to_select
        if self.features_to_select_k is None:
            self.features_to_select_k = self.dataframe.shape[1] - 1
        self.evaluation_metric = evaluation_metric

    def run_model_no_feature_selection(self, algorithm, model_name, train_dataframe, test_dataframe):
        train_dataframe = TabularDataset(train_dataframe)
        start_time_baseline = time.time()
        fitted_predictor = TabularPredictor(label=self.target_label,
                                            eval_metric=self.evaluation_metric,
                                            verbosity=1) \
            .fit(train_data=train_dataframe, hyperparameters={algorithm: {}})
        # Get the duration the baseline took
        baseline_duration = time.time() - start_time_baseline

        # Get the tuned hyperparameters
        training_results = fitted_predictor.info()
        hyperparameters = training_results['model_info'][model_name]['hyperparameters']

        # Get the performance and the feature importance given by the baseline model
        importance = fitted_predictor.feature_importance(data=test_dataframe, feature_stage='original')
        baseline_performance = fitted_predictor.evaluate(test_dataframe)[self.evaluation_metric]

        # print("Feature importance: " + importance)

        return hyperparameters, baseline_performance, baseline_duration

    def evaluate_all_models(self):
        # Print information about the dataset
        number_rows, number_columns = self.dataframe.shape
        print('Dataset: ' + self.dataset_name)
        print('Total columns: ' + str(number_columns - 1))
        print('Total rows: ' + str(number_rows))

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
            pearson_duration, spearman_duration, su_duration, ig_duration = \
            InML.feature_selection_select_k_best(current_train_dataframe,
                                                 self.target_label,
                                                 self.features_to_select_k)

        # LOOP: Go through each algorithm
        for algorithm, algorithm_name in self.algorithms_model_names.items():
            # COMPUTATION: Get the hyperparameters on the data set with all features
            hyperparameters, baseline_performance, baseline_duration = \
                self.run_model_no_feature_selection(algorithm, algorithm_name,
                                                    current_train_dataframe, current_test_dataframe)

            # LOOP: Go through each method
            correlation_methods = ['Pearson', 'Spearman', 'SU', 'IG']
            for ranked_features, correlation_method in zip([pearson_selected_features, spearman_selected_features,
                                                            su_selected_features, ig_selected_features],
                                                           correlation_methods):
                # LOOP: Go to all possible values of k (i.e. number of selected features)
                for subset_length in range(1, len(ranked_features) + 1):
                    # Get the current feature subset
                    current_subset = ranked_features[:subset_length]
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
                    elif correlation_method == 'Cramer':
                        total_duration += cramersv_duration
                    elif correlation_method == 'SU':
                        total_duration += su_duration
                    # Save the results to file
                    MLPipeline.write_to_file(dataset_name=self.dataset_name,
                                             algorithm_name=algorithm_name,
                                             correlation_method=correlation_method,
                                             subset_length=subset_length,
                                             current_subset=current_subset,
                                             current_performance=current_performance,
                                             current_duration=total_duration,
                                             baseline_performance=baseline_performance,
                                             baseline_duration=baseline_duration)


    @staticmethod
    def write_to_file(dataset_name, algorithm_name, correlation_method,
                      subset_length, current_subset, current_performance,
                      current_duration, baseline_performance, baseline_duration):
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
        file.write("CURRENT PERFORMANCE: " + str(current_performance) + '\n')
        file.write("CURRENT RUNTIME: " + str(current_duration) + '\n')
        file.write("BASELINE PERFORMANCE: " + str(baseline_performance) + '\n')
        file.write("BASELINE RUNTIME: " + str(baseline_duration) + '\n')
        file.write('\n')
        file.close()
