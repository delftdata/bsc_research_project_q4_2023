import pandas as pd
from warnings import filterwarnings
from sklearn.model_selection import train_test_split
from autogluon.features.generators import AutoMLPipelineFeatureGenerator, FillNaFeatureGenerator
from autogluon.tabular import TabularDataset
from autogluon.tabular import TabularPredictor
from .correlation_methods.pearson import PearsonFeatureSelection
from .correlation_methods.spearman import SpearmanFeatureSelection
from .correlation_methods.cramer import CramersVFeatureSelection
from .correlation_methods.su import SymmetricUncertaintyFeatureSelection

filterwarnings("ignore", category=UserWarning)

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
        self.number_of_features_to_select = 20
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

    def evaluate_model(self, algorithm, model_name, hyperparameters, train_dataframe, test_dataframe):
        train_data = TabularDataset(train_dataframe)
        fitted_predictor = TabularPredictor(label=self.target_label,
                                     eval_metric=self.evaluation_metric,
                                     verbosity=0) \
            .fit(train_data=train_data, hyperparameters={algorithm: hyperparameters})

        test_data = TabularDataset(test_dataframe)
        performance = fitted_predictor.evaluate(test_data)[self.evaluation_metric]
        print(performance)

    def evaluate_all_models(self):
        # Should we handle missing values?
        data_filled = FillNaFeatureGenerator().fit_transform(TabularDataset(self.dataframe))
        self.auxiliary_dataframe = AutoMLPipelineFeatureGenerator(
            enable_text_special_features=False,
            enable_text_ngram_features=False) \
            .fit_transform(data_filled)

        x_train, x_test, y_train, y_test = \
            train_test_split(self.auxiliary_dataframe.drop(columns=[self.target_label]),
                             self.auxiliary_dataframe[self.target_label],
                             test_size=0.2, random_state=1)
        train_dataframe = pd.concat([x_train, y_train], axis=1)
        test_dataframe = pd.concat([x_test, y_test], axis=1)

        # pearson_selected_features = PearsonFeatureSelection.feature_selection(train_dataframe,
        #                                                                       self.target_label,
        #                                                                       self.number_of_features_to_select)
        # pearson_selected_features.append(self.target_label)

        # spearman_selected_features = []
        # spearman_selected_features.append(self.target_label)

        cramersv_selected_features = CramersVFeatureSelection.feature_selection(train_dataframe,
                                                                                self.target_label,
                                                                                self.number_of_features_to_select)
        cramersv_selected_features.append(self.target_label)

        su_selected_features = \
            SymmetricUncertaintyFeatureSelection.feature_selection(train_dataframe,
                                                                   self.target_label,
                                                                   self.number_of_features_to_select)
        su_selected_features.append(self.target_label)

        # Compare the feature selection methods for each algorithm
        for algorithm, model_name in self.algorithms_model_names.items():
            print(algorithm)
            # Get the hyperparameters on the data set with all features
            hyperparameters = self.get_hyperparameters_no_feature_selection(algorithm, model_name)

            # Evaluate model on the features selected by the different correlation-based methods
            methods = ['Cramers V', 'SU']
            for feature_subset, method in zip([cramersv_selected_features, su_selected_features], methods):
                print(method)
                self.evaluate_model(algorithm=algorithm,
                                    hyperparameters=hyperparameters,
                                    model_name=model_name,
                                    train_dataframe=train_dataframe[feature_subset],
                                    test_dataframe=test_dataframe)


def evaluate_census_income_dataset():
    dataset_evaluator = DatasetEvaluator(
        dataset_file='../datasets/CensusIncome/CensusIncome.csv',
        dataset_name='CensusIncome',
        target_label='income_label',
        evaluation_metric='accuracy')

    dataset_evaluator.evaluate_all_models()


if __name__ == '__main__':
    # a = [1, 2, 3, 3]
    # b = [1000, 7, 6, 1000]
    # dataset_evaluator = DatasetEvaluator(
    #     dataset_file='../datasets/CensusIncome/CensusIncome.csv',
    #     dataset_name='CensusIncome',
    #     target_label='income_label',
    #     evaluation_metric='accuracy')
    # print(CramersVFeatureSelection.compute_correlation(a, b))
    # print(CramersVFeatureSelection.feature_selection(dataset_evaluator.auxiliary_dataframe,
    #       dataset_evaluator.target_label,5))
    # print(SpearmanFeatureSelection.compute_correlation(a, b))
    # print(CramersVFeatureSelection.compute_correlation(a, b))
    # print(SymmetricUncertaintyFeatureSelection.compute_correlation(a, b))

    evaluate_census_income_dataset()
