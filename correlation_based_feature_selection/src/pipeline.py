import pandas as pd
import csv
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer
from autogluon.features.generators import AutoMLPipelineFeatureGenerator, FillNaFeatureGenerator
from autogluon.tabular import TabularDataset
from autogluon.tabular import TabularPredictor
from .correlation_methods.pearson import PearsonFeatureSelection
from .correlation_methods.spearman import SpearmanFeatureSelection
from .correlation_methods.cramer import CramersVFeatureSelection
from .correlation_methods.su import SymmetricUncertaintyFeatureSelection
from .plots.number_of_features_plot import plot_over_number_of_features
from .encoding.encoding import OneHotEncoder
from .encoding.encoding import KBinsDiscretizer
from warnings import filterwarnings

filterwarnings("ignore", category=UserWarning)
filterwarnings("ignore", category=RuntimeWarning)
filterwarnings("ignore", category=FutureWarning)

class PreML:
    @staticmethod
    def imputation_most_common_value(dataframe):
        return dataframe.fillna(dataframe.mode(axis=1)[0])

    # All features will be continuous
    @staticmethod
    # TODO: what if target is nominal???
    def onehot_encoding(train_dataframe, test_dataframe, target_label):
        one_hot_encoder = OneHotEncoder()
        encoded_train_dataframe = one_hot_encoder.encode(train_dataframe, target_label)
        encoded_test_dataframe = one_hot_encoder.encode(test_dataframe, target_label)

        return encoded_train_dataframe, encoded_test_dataframe

    # All features will be nominal
    @staticmethod
    def k_bins_discretizer(train_dataframe, test_dataframe, target_label):
        # TODO: Should 'encode' be ordinal or onehot here?
        k_bins_discretizer = KBinsDiscretizer()
        encoded_train_dataframe = k_bins_discretizer.encode(train_dataframe, target_label)
        encoded_test_dataframe = k_bins_discretizer.encode(test_dataframe, target_label)

        return encoded_train_dataframe, encoded_test_dataframe


class InML:
    @staticmethod
    def feature_selection_select_k_best(train_dataframe, target_label, k):
        # Compute the ranking of features returned by each correlation method
        try:
            pearson_selected_features = PearsonFeatureSelection.feature_selection(train_dataframe, target_label, k)
        except ValueError:
            pearson_selected_features = []
        try:
            spearman_selected_features = SpearmanFeatureSelection.feature_selection(train_dataframe, target_label, k)
        except ValueError:
            spearman_selected_features = []
        cramersv_selected_features = CramersVFeatureSelection.feature_selection(train_dataframe, target_label, k)
        su_selected_features = SymmetricUncertaintyFeatureSelection.feature_selection(train_dataframe, target_label, k)

        return pearson_selected_features, spearman_selected_features, cramersv_selected_features, su_selected_features

    @staticmethod
    def feature_selection_select_above_t(train_dataframe, target_label, t):
        # TODO: implement this method
        pearson_selected_features = []
        spearman_selected_features = []
        cramersv_selected_features = []
        su_selected_features = []

        return pearson_selected_features, spearman_selected_features, cramersv_selected_features, su_selected_features


class PostML:
    @staticmethod
    def evaluate_model(algorithm, hyperparameters, train_dataframe, feature_subset,
                       target_label, test_dataframe, evaluation_metric):
        train_data = TabularDataset(train_dataframe[feature_subset])
        fitted_predictor = TabularPredictor(label=target_label,
                                            eval_metric=evaluation_metric,
                                            verbosity=0) \
            .fit(train_data=train_data, hyperparameters={algorithm: hyperparameters})

        # Evaluate the model with feature selection applied
        test_data = TabularDataset(test_dataframe)
        current_performance = fitted_predictor.evaluate(test_data)[evaluation_metric]

        return current_performance


class MLPipeline:
    def __init__(self, dataset_file, dataset_name, target_label, evaluation_metric):
        self.dataset_file = dataset_file
        self.dataset_name = dataset_name
        self.target_label = target_label
        self.dataframe = pd.read_csv(dataset_file)
        self.auxiliary_dataframe = pd.read_csv(dataset_file)
        # Specify the models to use: GBM (LightGBM), RF (RandomForest), LR (LinearModel), XGB (XGBoost)
        self.algorithms_model_names = {
            'GBM': 'LightGBM',
            'RF': 'RandomForest',
            'LR': 'LinearModel',
            'XGB': 'XGBoost'
        }
        # TODO: Experiment with other values for the number of features to select
        # TODO: Experiment with a threshold when selecting features
        self.features_to_select_k = self.dataframe.shape[1] - 1
        self.threshold_t = 0.5
        self.evaluation_metric = evaluation_metric

    def run_model_no_feature_selection(self, algorithm, model_name, train_dataframe, test_dataframe):
        fitted_predictor = TabularPredictor(label=self.target_label,
                                            eval_metric=self.evaluation_metric,
                                            verbosity=1) \
            .fit(train_data=train_dataframe, hyperparameters={algorithm: {}})

        # Get the tuned hyperparameters
        training_results = fitted_predictor.info()
        hyperparameters = training_results['model_info'][model_name]['hyperparameters']

        # Get the performance and the feature importance given by the baseline model
        importance = fitted_predictor.feature_importance(data=test_dataframe, feature_stage='original')
        baseline_performance = fitted_predictor.evaluate(test_dataframe)[self.evaluation_metric]

        # print("Feature importance: " + importance)

        return hyperparameters, baseline_performance

    def evaluate_all_models(self):
        # Prepare the data for Autogluon
        self.dataframe = TabularDataset(PreML.imputation_most_common_value(self.dataframe))
        self.dataframe = FillNaFeatureGenerator(inplace=True).fit_transform(self.dataframe)
        self.auxiliary_dataframe = AutoMLPipelineFeatureGenerator(
            enable_text_special_features=False,
            enable_text_ngram_features=False) \
            .fit_transform(self.dataframe)

        # Split the data into train and test
        x_train, x_test, y_train, y_test = \
            train_test_split(self.auxiliary_dataframe.drop(columns=[self.target_label]),
                             self.auxiliary_dataframe[self.target_label],
                             test_size=0.2, random_state=1)
        train_dataframe = pd.concat([x_train, y_train], axis=1)
        test_dataframe = pd.concat([x_test, y_test], axis=1)

        # Encode the features
        # all_continuous_train_dataframe, all_continuous_test_dataframe = \
        #     PreML.onehot_encoding(train_dataframe, test_dataframe, self.target_label)
        # all_nominal_train_dataframe, all_nominal_test_dataframe = \
        #     PreML.k_bins_discretizer(train_dataframe, test_dataframe, self.target_label)

        # 1 - normal, 2 - all continuous, 3 - all nominal
        dataset_type = 0
        # LOOP: Go through each type of data
        for current_train_dataframe, current_test_dataframe in zip(
                [train_dataframe], #, all_continuous_train_dataframe, all_nominal_train_dataframe],
                [test_dataframe]): # all_continuous_test_dataframe, all_nominal_test_dataframe]):

            dataset_type = dataset_type + 1

            # COMPUTATION: Compute the ranking of features returned by each correlation method
            pearson_selected_features, spearman_selected_features, cramersv_selected_features, su_selected_features = \
                InML.feature_selection_select_k_best(current_train_dataframe,
                                                     self.target_label,
                                                     self.features_to_select_k)

            # LOOP: Go through each algorithm
            for algorithm, algorithm_name in self.algorithms_model_names.items():
                # COMPUTATION: Get the hyperparameters on the data set with all features
                hyperparameters, baseline_performance = \
                    self.run_model_no_feature_selection(algorithm, algorithm_name,
                                                        current_train_dataframe, current_test_dataframe)

                # LOOP: Go through each method
                correlation_methods = ['Pearson', 'Spearman', 'Cramer\'s V', 'SU']
                correlation_methods_performances = []
                for ranked_features, correlation_method in zip([pearson_selected_features, spearman_selected_features,
                                                                cramersv_selected_features, su_selected_features],
                                                               correlation_methods):

                    correlation_method_performance = []
                    # LOOP: Go to all possible values of k (i.e. number of selected features)
                    for subset_length in range(1, len(ranked_features)):
                        # Get the current feature subset
                        current_subset = ranked_features[:subset_length]
                        current_subset.append(self.target_label)

                        current_performance = PostML.evaluate_model(algorithm=algorithm,
                                                                    hyperparameters=hyperparameters,
                                                                    train_dataframe=train_dataframe,
                                                                    feature_subset=current_subset,
                                                                    target_label=self.target_label,
                                                                    test_dataframe=current_test_dataframe,
                                                                    evaluation_metric=self.evaluation_metric)
                        correlation_method_performance.append(current_performance)

                        # Save the results to file
                        MLPipeline.write_to_file(self.dataset_name, str(dataset_type),
                                                 algorithm_name, correlation_method,
                                                 subset_length, current_subset,
                                                 current_performance, baseline_performance)
                    correlation_methods_performances.append(correlation_method_performance)

                plot_over_number_of_features(dataset_name=self.dataset_name,
                                             algorithm=algorithm_name,
                                             number_of_features=self.features_to_select_k,
                                             dataset_type=dataset_type,
                                             evaluation_metric=self.evaluation_metric,
                                             pearson_performance=correlation_methods_performances[0],
                                             spearman_performance=correlation_methods_performances[1],
                                             cramersv_performance=correlation_methods_performances[2],
                                             su_performance=correlation_methods_performances[3],
                                             baseline_performance=baseline_performance)

    @staticmethod
    def write_to_file(dataset_name, dataset_type, algorithm_name, correlation_method,
                      subset_length, current_subset, current_performance, baseline_performance):
        # Create the directory if it doesn't exist
        directory = "./results_tables"
        os.makedirs(directory, exist_ok=True)
        directory = "./results_tables/txt_files"
        os.makedirs(directory, exist_ok=True)
        directory = "./results_tables/csv_files"
        os.makedirs(directory, exist_ok=True)

        # Write the results to a txt file
        file_path = f"./results_tables/txt_files/{dataset_name}_{dataset_type}_{algorithm_name}_" \
                    f"{correlation_method}_{subset_length}.txt"
        file = open(file_path, "w")

        file.write("DATASET NAME: " + dataset_name + '\n')
        file.write("DATASET TYPE: " + str(dataset_type) + '\n')
        file.write("ALGORITHM NAME: " + algorithm_name + '\n')
        file.write("CORRELATION METHOD: " + correlation_method + '\n')
        file.write("SUBSET OF FEATURES: " + str(subset_length) + '\n')
        file.write("CURRENT FEATURE SUBSET: " + str(current_subset) + '\n')
        file.write("CURRENT PERFORMANCE: " + str(current_performance) + '\n')
        file.write("BASELINE PERFORMANCE: " + str(baseline_performance) + '\n')
        file.close()

        # Write the results to a csv file
        file_path = f"./results_tables/csv_files/{dataset_name}_{dataset_type}_{algorithm_name}_" \
                    f"{correlation_method}_{subset_length}.csv"

        with open(file_path, "w", newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["DATASET NAME", "DATASET TYPE",
                             "ALGORITHM NAME", "CORRELATION METHOD",
                             "SUBSET OF FEATURES", "CURRENT FEATURE SUBSET",
                             "CURRENT PERFORMANCE", "BASELINE PERFORMANCE"])
            writer.writerow([dataset_name, dataset_type, algorithm_name, correlation_method,
                             subset_length, current_subset, current_performance, baseline_performance])
