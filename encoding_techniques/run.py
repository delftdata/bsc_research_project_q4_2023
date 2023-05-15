import encoders.encoders as encoders
import pandas as pd
from models.models import AutogluonModel, SVMModel
import sys
import multiprocessing
n_jobs = multiprocessing.cpu_count()-1
import time

file = '../datasets/CensusIncome/CensusIncome.csv'
dataset_name = 'CensusIncome'
df = pd.read_csv(file)
label = 'income_label'
problemType = 'binary'

encoderNames = ['onehot', 'ordinal', 'target', 'catboost', 'count']

encoders = [encoders.OneHotEncoder(), encoders.OrdinalEncoder(
    ), encoders.TargetEncoder(), encoders.CatBoostEncoder(), encoders.CountEncoder()]


def runAutoGluonTests():
    algorithms = [
    {
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

    algorithmNames = [
        'LR', 'XGB',
        'GBM', 'RF']



    # encoderNames = ['target', 'catboost', 'count']

    # encoders = [encoders.TargetEncoder(), encoders.CatBoostEncoder(), encoders.CountEncoder()]

    # encoderName = ""
    # encoder = encoders.OneHotEncoder()

    # if sys.argv[1] == 'onehot':
    #     encoderName = 'onehot'
    #     encoder = encoders.OneHotEncoder()
    # elif sys.argv[1] == 'ordinal':
    #     encoderName = 'ordinal'
    #     encoder = encoders.OrdinalEncoder()
    # elif sys.argv[1] == 'target':
    #     encoderName = 'target'
    #     encoder = encoders.TargetEncoder()
    # elif sys.argv[1] == 'catboost':
    #     encoderName = 'catboost'
    #     encoder = encoders.CatBoostEncoder()
    # elif sys.argv[1] == 'count':
    #     encoderName = 'count'
    #     encoder = encoders.CountEncoder()

    for j, encoder in enumerate(encoders):
        encoderName = encoderNames[j]
        for i, algorithm in enumerate(algorithms):

            manualEncodingModel = AutogluonModel(
                problem_type=problemType, label=label, data_preprocessing=False, test_size=0.2, hyperparameters=algorithm)
            autoEncodingModel = AutogluonModel(
                problem_type=problemType, label=label, data_preprocessing=True, test_size=0.2, hyperparameters=algorithm)

            startTime = time.time()
            encoded_df = encoder.encode(df, label)
            encodeTime = time.time() - startTime
            manualEncodingModel.fit(encoded_df)
            durationManual = time.time() - startTime
            manualEncodingModel.evaluate(
                dataset_name, 'manual-' + algorithmNames[i], encoderName, durationManual, encodeTime)
            

            startTime = time.time()
            autoEncodingModel.fit(df)
            durationAuto = time.time() - startTime
            autoEncodingModel.evaluate(
                dataset_name, 'auto-' + algorithmNames[i], encoderName, durationAuto, 0)
        
param_grid = [
        {
        'C': [0.1, 10, 1000],
        'gamma': [1, 'scale'],
        'kernel': ['rbf', 'sigmoid', 'linear']
    },
    {
        'C': [0.1, 10, 1000],
        'gamma': [1, 'scale'],
        'kernel': ['poly'],
        'degree': [6, 7, 8, 9]
    }
]


svm_hyperparameters = [{
        'C': 10,
        'gamma': 1,
        'kernel': 'poly',
        'degree': 8
    },
    {
        'C': 10,
        'gamma': 1,
        'kernel': 'poly',
        'degree': 8
    },
    {
        'C': 10,
        'gamma': 1,
        'kernel': 'poly',
        'degree': 8
    },
    {
        'C': 10,
        'gamma': 1,
        'kernel': 'poly',
        'degree': 8
    },
    {
        'C': 1000,
        'gamma': 'scale',
        'kernel': 'poly',
        'degree': 9
    },
    ]
def run_svm_grid():
    for j, encoder in enumerate(encoders):
        svmModel = SVMModel(problem_type=problemType, label=label,
                        data_preprocessing=False, test_size=0.2)

        svmModel.grid_search(encoder.encode(df.head(), label), dataset_name, encoderNames[j], param_grid)


    
def run_svm_model():
    for j, encoder in enumerate(encoders):
        svmModel = SVMModel(problem_type=problemType, label=label,
                        data_preprocessing=False, test_size=0.2, hyperparameters = svm_hyperparameters[j])

        startTime = time.time()
        encoded_df = encoder.encode(df, label)
        encodeTime = time.time() - startTime

        svmModel.fit(encoded_df)
        svmTime = time.time() - startTime
        
        svmModel.evaluate(dataset_name, encoderNames[j], svmTime, encodeTime)


# run_svm_grid()
run_svm_model()