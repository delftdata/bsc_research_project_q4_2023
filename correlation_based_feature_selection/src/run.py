import pandas as pd
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
from sklearn.impute import SimpleImputer

import numpy as np

class DatasetEvaluator:
    def __init__(self, dataset_file, dataset_name, target_label, evaluation_metric):
        self.dataset_file = dataset_file
        self.dataset_name = dataset_name
        self.target_label = target_label
        self.dataframe = pd.read_csv(dataset_file)
        self.auxiliary_dataframe = pd.read_csv(dataset_file)
        # Specify the models to use
        # GBM (LightGBM), RF (RandomForest), LR (LinearModel), XGB (XGBoost)
        self.algorithms_model_names = {
            'GBM': 'LightGBM',
            # 'RF': 'RandomForest',
            # 'LR': 'LinearModel',
            # 'XGB': 'XGBoost'
        }
        # TODO: Experiment with other values for the number of features to select
        # TODO: Experiment with a threshold when selecting features
        self.number_of_features_to_select = self.dataframe.shape[1] - 1
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

        print("Feature importance")
        print(importance)
        print(" ")
        print("Baseline performance")
        print(baseline_performance)
        print(" ")

        return hyperparameters, baseline_performance

    def evaluate_model_varying_features(self, algorithm, hyperparameters, train_dataframe,
                                        feature_subset, target_label, test_dataframe):
        performances = []

        for subset_length in range(1, len(feature_subset)):
            print(subset_length)
            # Get the current feature subset
            current_subset = feature_subset[:subset_length]
            current_subset.append(target_label)
            print(current_subset)

            train_data = TabularDataset(train_dataframe[current_subset])
            fitted_predictor = TabularPredictor(label=self.target_label,
                                         eval_metric=self.evaluation_metric,
                                         verbosity=0) \
                .fit(train_data=train_data, hyperparameters={algorithm: hyperparameters})

            # Evaluate the model with feature selection applied
            test_data = TabularDataset(test_dataframe)
            current_performance = fitted_predictor.evaluate(test_data)[self.evaluation_metric]
            print(current_performance)
            print('\n')
            performances.append(current_performance)

        return performances

    # TODO: what if target is nominal???
    def transform_all_features_continuous(self, train_dataframe, test_dataframe):
        one_hot_encoder = OneHotEncoder()
        encoded_train_dataframe = one_hot_encoder.encode(train_dataframe, self.target_label)
        encoded_test_dataframe = one_hot_encoder.encode(test_dataframe, self.target_label)

        return encoded_train_dataframe, encoded_test_dataframe

    def transform_all_features_nominal(self, train_dataframe, test_dataframe):
        # TODO: Should 'encode' be ordinal or onehot here?
        k_bins_discretizer = KBinsDiscretizer()
        encoded_train_dataframe = k_bins_discretizer.encode(train_dataframe, self.target_label)
        encoded_test_dataframe = k_bins_discretizer.encode(test_dataframe, self.target_label)

        return encoded_train_dataframe, encoded_test_dataframe

    def impute_most_frequent(self, df):
        imputer = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
        imputer.set_output(transform="pandas")

        for i, column in enumerate(df.columns):
            if any(df.iloc[:, [i]].isna().values):
                df[column] = imputer.fit_transform(df.iloc[:, [i]])

        return df

    def evaluate_all_models(self):
        self.dataframe = TabularDataset(self.dataframe.fillna(self.dataframe.mode(axis=1)[0]))
        #self.dataframe = self.impute_most_frequent(self.dataframe)
        #self.dataframe = self.dataframe.dropna(axis=0, how="any")
        self.dataframe = FillNaFeatureGenerator(inplace=True).fit_transform(self.dataframe)
        self.auxiliary_dataframe = AutoMLPipelineFeatureGenerator(
            enable_text_special_features=False,
            enable_text_ngram_features=False) \
            .fit_transform(self.dataframe)

        x_train, x_test, y_train, y_test = \
            train_test_split(self.auxiliary_dataframe.drop(columns=[self.target_label]),
                             self.auxiliary_dataframe[self.target_label],
                             test_size=0.2, random_state=1)
        train_dataframe = pd.concat([x_train, y_train], axis=1)
        test_dataframe = pd.concat([x_test, y_test], axis=1)

        # Encode the features
        all_continuous_train_dataframe, all_continuous_test_dataframe = self. \
            transform_all_features_continuous(train_dataframe, test_dataframe)
        print('HELLO')
        print(all_continuous_train_dataframe)

        all_nominal_train_dataframe, all_nominal_test_dataframe = self. \
            transform_all_features_nominal(train_dataframe, test_dataframe)

        # Go through each type of data
        for current_train_dataframe, current_test_dataframe in zip(
                [all_continuous_train_dataframe, all_nominal_train_dataframe],
                [all_continuous_test_dataframe, all_nominal_test_dataframe]):

            # Compute the ranking of features returned by each correlation method
            pearson_selected_features = PearsonFeatureSelection.feature_selection(current_train_dataframe,
                                                                                  self.target_label,
                                                                                  self.number_of_features_to_select)

            spearman_selected_features = SpearmanFeatureSelection.feature_selection(current_train_dataframe,
                                                                                    self.target_label,
                                                                                    self.number_of_features_to_select)

            cramersv_selected_features = CramersVFeatureSelection.feature_selection(current_train_dataframe,
                                                                                    self.target_label,
                                                                                    self.number_of_features_to_select)

            su_selected_features = \
                SymmetricUncertaintyFeatureSelection.feature_selection(current_train_dataframe,
                                                                       self.target_label,
                                                                       self.number_of_features_to_select)

            # Go through each algorithm
            for algorithm, model_name in self.algorithms_model_names.items():
                print(algorithm)
                # Get the hyperparameters on the data set with all features
                hyperparameters, baseline_performance = \
                    self.run_model_no_feature_selection(algorithm, model_name,
                                                        current_train_dataframe, current_test_dataframe)

                # Evaluate model on the features selected by the different correlation-based methods
                methods = ['Pearson', 'Spearman', 'Cramer\'s V', 'SU']
                all_performances_methods = []

                # Go through each method
                for feature_subset, method in zip([pearson_selected_features, spearman_selected_features,
                                                   cramersv_selected_features, su_selected_features], methods):
                    print(method)
                    algorithm_performances = self.evaluate_model_varying_features(algorithm=algorithm,
                                                                                  hyperparameters=hyperparameters,
                                                                                  train_dataframe=current_train_dataframe,
                                                                                  feature_subset=feature_subset,
                                                                                  target_label=self.target_label,
                                                                                  test_dataframe=current_test_dataframe)
                    all_performances_methods.append(algorithm_performances)

                plot_over_number_of_features(algorithm=algorithm, dataset_name=self.dataset_name,
                                             number_of_features=self.number_of_features_to_select,
                                             evaluation_metric=self.evaluation_metric,
                                             pearson_performance=all_performances_methods[0],
                                             spearman_performance=all_performances_methods[1],
                                             cramersv_performance=all_performances_methods[2],
                                             su_performance=all_performances_methods[3],
                                             baseline_performance=baseline_performance)


def evaluate_census_income_dataset():
    dataset_evaluator = DatasetEvaluator(
        dataset_file='../datasets/CensusIncome/CensusIncome.csv',
        dataset_name='CensusIncome',
        target_label='income_label',
        evaluation_metric='accuracy')

    dataset_evaluator.evaluate_all_models()


def evaluate_breast_cancer_dataset():
    dataset_evaluator = DatasetEvaluator(
        dataset_file='../datasets/breast-cancer/data.csv',
        dataset_name='BreastCancer',
        target_label='diagnosis',
        evaluation_metric='accuracy')

    dataset_evaluator.evaluate_all_models()


def evaluate_steel_plates_fault_dataset():
    dataset_evaluator = DatasetEvaluator(
        dataset_file='../datasets/steel-plates-faults/steel_faults_train.csv',
        dataset_name='SteelPlatesFaults',
        target_label='Class',
        evaluation_metric='accuracy')

    # column_types = dataset_evaluator.dataframe.dtypes.value_counts()
    # print(column_types)
    #print(len(getContinuousColumns(dataset_evaluator.dataframe)))
    dataset_evaluator.evaluate_all_models()
    # Assuming you have a dataframe called 'df'
    # rows_with_nan = dataset_evaluator.dataframe[dataset_evaluator.dataframe.isna().any(axis=1)]
    # print(dataset_evaluator.loc[rows_with_nan.index])


def evaluate_connect4_dataset():
    dataset_evaluator = DatasetEvaluator(
        dataset_file='../datasets/Connect-4/Connect-4.csv',
        dataset_name='Connect4',
        target_label='winner',
        evaluation_metric='accuracy')

    # nan_columns = dataset_evaluator.auxiliary_dataframe.columns[dataset_evaluator.auxiliary_dataframe.isna().any()].tolist()
    #
    # # Find columns with infinite (inf) values
    # inf_columns = dataset_evaluator.auxiliary_dataframe.columns[np.isinf(dataset_evaluator.auxiliary_dataframe).any()].tolist()
    #
    # print(nan_columns)
    # print(inf_columns)
    #
    # nan_rows = dataset_evaluator.auxiliary_dataframe[dataset_evaluator.auxiliary_dataframe.isna().any(axis=1)]
    # print(nan_rows)

    dataset_evaluator.evaluate_all_models()

    # column_types = dataset_evaluator.dataframe.dtypes.value_counts()
    # print(column_types)


def evaluate_housing_prices_dataset():
    dataset_evaluator = DatasetEvaluator(
        dataset_file='../datasets/housing-prices/train.csv',
        dataset_name='Housing Prices',
        target_label='SalePrice',
        evaluation_metric='accuracy')

    # column_types = dataset_evaluator.dataframe.dtypes.value_counts()
    # print(column_types)
    dataset_evaluator.evaluate_all_models()


if __name__ == '__main__':
    # evaluate_census_income_dataset()
    # evaluate_breast_cancer_dataset()
    evaluate_steel_plates_fault_dataset()
    #evaluate_connect4_dataset()
    #evaluate_housing_prices_dataset()
