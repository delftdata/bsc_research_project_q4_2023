import pandas as pd
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
            'GBM': 'LightGBM',
            'RF': 'RandomForest',
            'LR': 'LinearModel',
            'XGB': 'XGBoost'
        }
        self.evaluation_metric = evaluation_metric

    def get_hyperparameters_no_feature_selection(self, algorithm, model_name):
        predictor = AutogluonModel(dataframe=self.auxiliary_dataframe,
                                   target_label=self.target_label,
                                   evaluation_metric=self.evaluation_metric,
                                   test_size=0.0,
                                   hyperparameters={algorithm: {}})
        fitting_results = predictor.fit()

        # Get tuned hyperparameters
        return fitting_results['model_info'][model_name]['hyperparameters']

    def evaluate_model(self, algorithm, model_name, hyperparameters):
        predictor = AutogluonModel(dataframe=self.auxiliary_dataframe,
                                   target_label=self.target_label,
                                   evaluation_metric=self.evaluation_metric,
                                   test_size=0.2,
                                   hyperparameters={algorithm: hyperparameters})
        fitting_results = predictor.fit()


    def evaluate_all_models(self):
        for algorithm, model_name in self.algorithms_model_names.items():
            # Get the hyperparameters on the data set with all features
            hyperparameters = self.get_hyperparameters_no_feature_selection(algorithm, model_name)
            self.evaluate_model(algorithm, model_name, hyperparameters)


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
