import pandas as pd
from sklearn.model_selection import train_test_split
from autogluon.features.generators import AutoMLPipelineFeatureGenerator, FillNaFeatureGenerator
from autogluon.tabular import TabularDataset
from autogluon.tabular import TabularPredictor
from .correlation_methods.pearson import PearsonFeatureSelection
from .correlation_methods.spearman import SpearmanFeatureSelection
from .correlation_methods.cramer import CramersVFeatureSelection
from .correlation_methods.su import SymmetricUncertaintyFeatureSelection
from .plots.number_of_features_plot import plot_over_number_of_features
from .encoding.encoding import OneHotEncoder


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
            'RF': 'RandomForest',
            'LR': 'LinearModel',
            'XGB': 'XGBoost'
        }
        # TODO: Experiment with other values for the number of features to select
        self.number_of_features_to_select = self.dataframe.shape[1] - 1
        self.evaluation_metric = evaluation_metric

    def get_hyperparameters_no_feature_selection(self, algorithm, model_name):
        train_data = TabularDataset(self.auxiliary_dataframe)
        fitted_predictor = TabularPredictor(label=self.target_label,
                                     eval_metric=self.evaluation_metric,
                                     verbosity=0) \
            .fit(train_data=train_data, hyperparameters={algorithm: {}})

        # Get tuned hyperparameters
        training_results = fitted_predictor.info()
        return training_results['model_info'][model_name]['hyperparameters']

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

    def evaluate_all_models(self):
        # TODO: Should we handle missing values?
        filled_dataframe = FillNaFeatureGenerator().fit_transform(TabularDataset(self.dataframe))
        self.auxiliary_dataframe = AutoMLPipelineFeatureGenerator(
            enable_text_special_features=False,
            enable_text_ngram_features=False) \
            .fit_transform(filled_dataframe)

        x_train, x_test, y_train, y_test = \
            train_test_split(self.auxiliary_dataframe.drop(columns=[self.target_label]),
                             self.auxiliary_dataframe[self.target_label],
                             test_size=0.2, random_state=1)
        train_dataframe = pd.concat([x_train, y_train], axis=1)
        test_dataframe = pd.concat([x_test, y_test], axis=1)

        one_hot_encoder = OneHotEncoder()
        dataframe_not_categorical = one_hot_encoder.encode(train_dataframe, self.target_label)
        pearson_selected_features = PearsonFeatureSelection.feature_selection(train_dataframe,
                                                                              self.target_label,
                                                                              self.number_of_features_to_select)

        spearman_selected_features = SpearmanFeatureSelection.feature_selection(train_dataframe,
                                                                                self.target_label,
                                                                                self.number_of_features_to_select)

        cramersv_selected_features = CramersVFeatureSelection.feature_selection(train_dataframe,
                                                                                self.target_label,
                                                                                self.number_of_features_to_select)

        su_selected_features = \
            SymmetricUncertaintyFeatureSelection.feature_selection(train_dataframe,
                                                                   self.target_label,
                                                                   self.number_of_features_to_select)

        # Compare the feature selection methods for each algorithm
        for algorithm, model_name in self.algorithms_model_names.items():
            print(algorithm)
            # Get the hyperparameters on the data set with all features
            hyperparameters = self.get_hyperparameters_no_feature_selection(algorithm, model_name)

            # TODO: Add all methods (+ figure out the necessary encoding)
            # Evaluate model on the features selected by the different correlation-based methods
            methods = ['Pearson', 'Spearman', 'Cramer\'s V', 'SU']
            all_performances_methods = []
            for feature_subset, method in zip([pearson_selected_features, spearman_selected_features,
                                               cramersv_selected_features, su_selected_features], methods):
                print(method)
                algorithm_performances = self.evaluate_model_varying_features(algorithm=algorithm,
                                                                              hyperparameters=hyperparameters,
                                                                              train_dataframe=train_dataframe,
                                                                              feature_subset=feature_subset,
                                                                              target_label=self.target_label,
                                                                              test_dataframe=test_dataframe)
                all_performances_methods.append(algorithm_performances)

            plot_over_number_of_features(algorithm=algorithm, number_of_features=self.number_of_features_to_select,
                                         evaluation_metric=self.evaluation_metric,
                                         pearson_performance=all_performances_methods[0],
                                         spearman_performance=all_performances_methods[1],
                                         cramersv_performance=all_performances_methods[2],
                                         su_performance=all_performances_methods[3])


def evaluate_census_income_dataset():
    dataset_evaluator = DatasetEvaluator(
        dataset_file='../datasets/CensusIncome/CensusIncome.csv',
        dataset_name='CensusIncome',
        target_label='income_label',
        evaluation_metric='accuracy')

    # TODO: Pearson and Spearman will fail unless encoding is done
    dataset_evaluator.evaluate_all_models()


def evaluate_breast_cancer_dataset():
    dataset_evaluator = DatasetEvaluator(
        dataset_file='../datasets/breast-cancer/data.csv',
        dataset_name='BreastCancer',
        target_label='diagnosis',
        evaluation_metric='accuracy')

    dataset_evaluator.evaluate_all_models()


if __name__ == '__main__':
    evaluate_census_income_dataset()
    # evaluate_breast_cancer_dataset()
