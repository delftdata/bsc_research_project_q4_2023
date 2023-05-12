import encoders.encoders as encoders
import pandas as pd
from models.models import AutogluonModel, SVMModel

file = '../datasets/CensusIncome/CensusIncome.csv'
dataset_name = 'CensusIncome'
df = pd.read_csv(file)
label = 'income_label'
problemType = 'binary'

algorithms = [{
    'LR': {}
},
    {
    'XGB': {},
},
    {
    'GBM': {},
},
    {
    'RF': {},
}]

encoders = [encoders.OneHotEncoder(), encoders.OrdinalEncoder(
), encoders.TargetEncoder(), encoders.CatBoostEncoder(), encoders.CatBoostEncoder()]

for algorithm in algorithms:
    manualEncodingModel = AutogluonModel(
        problem_type=problemType, label=label, data_preprocessing=False, test_size=0.2, hyperparameters=algorithm)
    autoEncodingModel = AutogluonModel(
        problem_type=problemType, label=label, data_preprocessing=True, test_size=0.2, hyperparameters=algorithm)
    
    for encoder in encoders:
        manualEncodingModel.fit(encoders.OneHotEncoder().encode(df, label))
        manualEncodingModel.evaluate()
        autoEncodingModel.fit(encoders.OneHotEncoder().encode(df, label))
        autoEncodingModel.evaluate()


svmModel = SVMModel(problem_type='binary', label=label,
                    data_preprocessing=False, test_size=0.2)
param_grid = [
    #     {
    #     'C': [0.1, 1, 10, 100, 1000],
    #     'gamma': [1, 0.1, 0.01, 0.001, 'scale', 'auto'],
    #     'kernel': ['rbf', 'sigmoid', 'Autogluon']
    # },
    # {
    #     'C': [0.1, 1, 10, 100, 1000],
    #     'gamma': [1, 0.1, 0.01, 0.001, 'scale', 'auto'],
    #     'kernel': ['poly'],
    #     'degree': [4, 5, 6, 7]
    # }
    {
        'C': [0.1, 1, 10, 100, 1000],
        'gamma': [1, 0.1, 0.01, 0.001, 'scale', 'auto'],
        'kernel': ['poly'],
        'degree': [6, 7, 8, 9]
    }
]
file = 'grid_searches/' +\
    str(dataset_name) + '_svm_hyperparameters' + '.txt'

svmModel.grid_search(encoders.CountEncoder().encode(
    df.head(100), label), param_grid, file)
# svmModel.fit(encoders.OneHotEncoder().encode(df.head(10000), label))
# print(svmModel.evaluate())
