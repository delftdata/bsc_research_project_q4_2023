from .correlation_methods.pearson import PearsonFeatureSelection
from .correlation_methods.spearman import SpearmanFeatureSelection
from .correlation_methods.cramer import CramersVFeatureSelection
from .correlation_methods.su import SymmetricUncertaintyFeatureSelection
from .models.models import AutogluonModel, SVMModel

import pandas as pd

if __name__ == '__main__':
    a = [1, 2, 3, 3]
    b = [1000, 7, 6, 1000]
    # a = ['a', 'b', 'c', 'a', 'a', 'b']
    # b = ['ddfdd', 'dsdfse', 'sdfsdf', 'sdfsf', 'coobbbi', 'coobbbi']
    print(PearsonFeatureSelection.compute_correlation(a, b))
    print(SpearmanFeatureSelection.compute_correlation(a, b))
    print(CramersVFeatureSelection.compute_correlation(a, b))
    print(SymmetricUncertaintyFeatureSelection.compute_correlation(a, b))

def evaluate_census_income_dataset():
    dataset_file = 'datasets/CensusIncome/CensusIncome.csv'
    dataset_name = 'CensusIncome'
    dataframe = pd.read_csv(dataset_file)
    label = 'income_label'

    # specify the models to use:
    # GBM (LightGBM), RF (random forest), LR (linear regression), XGB (XGBoost)
    hyper_parameters = {'RF': {}, 'GBM': {}, 'XGB': {}, 'LR': {}}

    autogluonModel = AutogluonModel(
        problem_type='binary', label=label, data_preprocessing=False, test_size=0.2, hyperparameters=hyper_parameters)
    # AutogluonModel.fit(encoders.OneHotEncoder().encode(df.head(10000), label))
    # print(AutogluonModel.evaluate())
