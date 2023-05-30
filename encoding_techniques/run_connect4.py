import encoders.encoders as encoders
import pandas as pd
from models.models import AutogluonModel, SVMModel
import multiprocessing
n_jobs = multiprocessing.cpu_count()-1
import time

file = '../datasets/Connect4/connect4.csv'
dataset_name = 'Connect4'
df = pd.read_csv(file)
label = 'label'
problemType = 'multiclass'

df[label] = df[label].replace('win', 2)
df[label] = df[label].replace('draw', 1)
df[label] = df[label].replace('loss', 0)
df[label] = df[label].astype('int')

# print(df.head(2))

encoderNames = ['autogluon', 'onehot', 'ordinal', 'target', 'catboost', 'count']

encoders_list = [encoders.AutoGluonEncoder(), encoders.OneHotEncoder(), encoders.OrdinalEncoder(
    ), encoders.TargetEncoder(), encoders.CatBoostEncoder(), encoders.CountEncoder()]

encoded_dataframes = [encoders.AutoGluonEncoder().encode(df, label), encoders.OneHotEncoder().encode(df, label), encoders.OrdinalEncoder(
    ).encode(df, label), encoders.TargetEncoder().encode(df, label), encoders.CatBoostEncoder().encode(df, label), encoders.CountEncoder().encode(df, label)]



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

    
    for j, encoder in enumerate(encoders_list):
        startTime = time.time()
        encoded_df = encoder.encode(df, label)
        encodeTime = time.time() - startTime
        for i, algorithm in enumerate(algorithms):

            manualEncodingModel = AutogluonModel(
                problem_type=problemType, label=label, data_preprocessing=False, test_size=0.2, hyperparameters=algorithm)

            manualEncodingModel.fit(encoded_df)
            durationManual = time.time() - startTime
            manualEncodingModel.evaluate(
                dataset_name,  algorithmNames[i], encoderNames[j], durationManual, encodeTime)
    
        
param_grid = [
        {
        'C': [0.1, 100, 10, 1000],
        'gamma': [1, 'scale'],
        'kernel': ['rbf', 'sigmoid', 'linear']
    },{
    
        'C': [0.1, 10, 1000],
        'gamma': [1, 'scale'],
        'kernel': ['poly'],
        'degree': [6, 7, 8, 9]
    }]


# ['autogluon', 'onehot', 'ordinal', 'target', 'catboost', 'count']
svm_hyperparameters = [{
        'C': 1000,
        'gamma': 'scale',
        'kernel': 'poly',
        'degree': 6
    },
    {
        'C': 1000,
        'gamma': 'scale',
        'kernel': 'poly',
        'degree': 9
    },
    {
        'C': 1000,
        'gamma': 'scale',
        'kernel': 'poly',
        'degree': 9
    },
    {
        'C': 1000,
        'gamma': 'scale',
        'kernel': 'poly',
        'degree': 6
    },
    {
        'C': 1000,
        'gamma': 'scale',
        'kernel': 'poly',
        'degree': 6
    },
    {
        'C': 0.1,
        'gamma': 'scale',
        'kernel': 'poly',
        'degree': 6
    },
    ]
def run_svm_grid():
    
        svmModel = SVMModel(problem_type=problemType, label=label,
                        data_preprocessing=False, test_size=0.2)

        svmModel.grid_search(df, dataset_name, encoderNames[j], param_grid)



# max_rows = 60000
# encoded_dataframes = [ encoders.AutoGluonEncoder().encode(df.head(max_rows), label), encoders.OneHotEncoder().encode(df.head(max_rows), label), encoders.OrdinalEncoder().encode(df.head(max_rows), label)]

def run_svm_model():
    for j, df in enumerate(encoded_dataframes):
        print(f'Running encoder {encoderNames[j]}')
        svmModel = SVMModel(problem_type=problemType, label=label,
                        data_preprocessing=False, test_size=0.2, hyperparameters = svm_hyperparameters[j])

        startTime = time.time()
        # encoded_df = encoder.encode(df, label)
        # encodeTime = time.time() - startTime

        svmModel.fit(df)
        svmTime = time.time() - startTime
        
        svmModel.evaluate(dataset_name, encoderNames[j], svmTime, 0)



# runAutoGluonTests()
# run_svm_grid()
run_svm_model()

