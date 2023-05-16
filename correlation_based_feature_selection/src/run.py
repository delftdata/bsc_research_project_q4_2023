import pandas as pd
from sklearn.model_selection import train_test_split
from autogluon.features.generators import AutoMLPipelineFeatureGenerator
from autogluon.tabular import TabularDataset
from .correlation_methods.pearson import PearsonFeatureSelection
from .correlation_methods.spearman import SpearmanFeatureSelection
from .correlation_methods.cramer import CramersVFeatureSelection
from .correlation_methods.su import SymmetricUncertaintyFeatureSelection
from .models.models import AutogluonModel

class DatasetEvaluator:
    def __init__(self, dataset_file, dataset_name, target_label, evaluation_metric):
        self.dataset_file = dataset_file
        self.dataset_name = dataset_name
        self.target_label = target_label
        self.dataframe = pd.read_csv(dataset_file)
        self.auxiliary_dataframe = AutoMLPipelineFeatureGenerator(
            enable_text_special_features=False,
            enable_text_ngram_features=False)\
            .fit_transform(X=TabularDataset(self.dataframe))
        # Specify the models to use
        # GBM (LightGBM), RF (RandomForest), LR (LinearModel), XGB (XGBoost)
        self.algorithms_model_names = {
            #'GBM': 'LightGBM',
            #'RF': 'RandomForest',
            #'LR': 'LinearModel',
            'XGB': 'XGBoost'
        }
        self.number_of_features_to_select = 20
        self.evaluation_metric = evaluation_metric

    def get_hyperparameters_no_feature_selection(self, algorithm, model_name):
        predictor = AutogluonModel(algorithm=algorithm, model_name=model_name,
                                   target_label=self.target_label,
                                   evaluation_metric=self.evaluation_metric,
                                   hyperparameters={})
        # Get tuned hyperparameters
        return predictor.get_hyperparameters(self.auxiliary_dataframe)

    def evaluate_model(self, algorithm, model_name, hyperparameters, df_train, df_test):
        predictor = AutogluonModel(algorithm=algorithm, model_name=model_name,
                                   target_label=self.target_label,
                                   evaluation_metric=self.evaluation_metric,
                                   hyperparameters=hyperparameters)
        fitting_results = predictor.fit(df_train)

    def evaluate_all_models(self):
        x_train, x_test, y_train, y_test = \
            train_test_split(self.auxiliary_dataframe.drop(columns=[self.target_label]),
                             self.auxiliary_dataframe[self.target_label],
                             test_size=0.2, random_state=1)
        train_dataframe = pd.concat([x_train, y_train], axis=1)
        test_dataframe = pd.concat([x_test, y_test], axis=1)
        train_data = TabularDataset(train_dataframe)

        for algorithm, model_name in self.algorithms_model_names.items():
            # Get the hyperparameters on the data set with all features
            hyperparameters = self.get_hyperparameters_no_feature_selection(algorithm, model_name)
            # TODO
            self.evaluate_model(algorithm, hyperparameters, model_name, 1, 1)


def evaluate_census_income_dataset():
    dataset_evaluator = DatasetEvaluator(
        dataset_file='../datasets/CensusIncome/CensusIncome.csv',
        dataset_name='CensusIncome',
        target_label='income_label',
        evaluation_metric='accuracy')

    print(dataset_evaluator.get_hyperparameters_no_feature_selection('XGB', 'XGBoost'))


if __name__ == '__main__':
    # a = [1, 2, 3, 3]
    # b = [1000, 7, 6, 1000]
    # print(PearsonFeatureSelection.compute_correlation(a, b))
    # print(SpearmanFeatureSelection.compute_correlation(a, b))
    # print(CramersVFeatureSelection.compute_correlation(a, b))
    # print(SymmetricUncertaintyFeatureSelection.compute_correlation(a, b))

    evaluate_census_income_dataset()
