import pandas as pd
from .correlation_methods.pearson import PearsonFeatureSelection
from .correlation_methods.spearman import SpearmanFeatureSelection
from .correlation_methods.cramer import CramersVFeatureSelection
from .correlation_methods.su import SymmetricUncertaintyFeatureSelection
from .models.models import AutogluonModel


class DatasetEvaluator:
    def __init__(self, dataset_file, dataset_name, target_label):
        self.dataset_file = dataset_file
        self.dataset_name = dataset_name
        self.target_label = target_label
        self.dataframe = pd.read_csv(dataset_file)
        # specify the models to use; choose between:
        # GBM (LightGBM), RF (random forest), LR (linear regression), XGB (XGBoost)
        self.hyperparameters = {'RF': {}, 'GBM': {}, 'XGB': {}, 'LR': {}}


def evaluate_census_income_dataset():
    dataset_evaluator = DatasetEvaluator(
        '../datasets/CensusIncome/CensusIncome.csv', 'CensusIncome', 'income_label')

    autogluon_model = AutogluonModel(
        problem_type='binary', label=dataset_evaluator.target_label, data_preprocessing=False,
        test_size=0.2, hyperparameters=dataset_evaluator.hyperparameters)
    autogluon_model.fit(dataset_evaluator.dataframe)
    print(autogluon_model.evaluate())


if __name__ == '__main__':
    a = [1, 2, 3, 3]
    b = [1000, 7, 6, 1000]
    print(PearsonFeatureSelection.compute_correlation(a, b))
    print(SpearmanFeatureSelection.compute_correlation(a, b))
    print(CramersVFeatureSelection.compute_correlation(a, b))
    print(SymmetricUncertaintyFeatureSelection.compute_correlation(a, b))

    evaluate_census_income_dataset()
