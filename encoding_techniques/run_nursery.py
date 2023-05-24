import encoders.encoders as encoders
import pandas as pd
from models.models import AutogluonModel, SVMModel
import multiprocessing
n_jobs = multiprocessing.cpu_count()-1
import time



file = '../datasets/Nursery/nursery.csv'
dataset_name = 'Nursery'
df = pd.read_csv(file)

# print(df.columns)

label = 'label'
problemType = 'multiclass'
df = df.fillna(0)

df[label] = df[label].replace('not_recom', 0)
df[label] = df[label].replace('spec_prior', 1)
df[label] = df[label].replace('recommend', 2)
df[label] = df[label].replace('very_recom', 3)
df[label] = df[label].replace('priority', 4)

# print(df[label].unique())

df[label] = df[label].astype('int')

encoderNames = ['autogluon', 'onehot', 'ordinal', 'target', 'catboost', 'count']

encoders_list = [encoders.AutoGluonEncoder(), encoders.OneHotEncoder(), encoders.OrdinalEncoder(
    ), encoders.TargetEncoder(), encoders.CatBoostEncoder(), encoders.CountEncoder()]

encoded_dataframes = [encoders.AutoGluonEncoder().encode(df, label), encoders.OneHotEncoder().encode(df, label), encoders.OrdinalEncoder(
    ).encode(df, label), encoders.TargetEncoder().encode(df, label), encoders.CatBoostEncoder().encode(df, label), encoders.CountEncoder().encode(df, label)]


# encoderNames = ['count']

# encoders_list = [encoders.CountEncoder()]

# encoded_dataframes = [encoders.CountEncoder().encode(df, label)]

# print(encoded_dataframes[0].head(5))
# print(encoded_dataframes[0]['housing'].unique())
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
svm_hyperparameters = [
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
        'C': 1000,
        'gamma': 'scale',
        'kernel': 'poly',
        'degree': 6
    },
    {
        'C': 0.1,
        'gamma': 'scale',
        'kernel': 'poly',
        'degree': 8
    },
    {
        'C': 10,
        'gamma': 'scale',
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
    for j, df in enumerate(encoded_dataframes):
        svmModel = SVMModel(problem_type=problemType, label=label,
                        data_preprocessing=False, test_size=0.2)

        svmModel.grid_search(df, dataset_name, encoderNames[j], param_grid)


    
def run_svm_model():
    for j, df in enumerate(encoded_dataframes):
        print(f'Running encoder {encoderNames[j]}')
        svmModel = SVMModel(problem_type=problemType, label=label,
                        data_preprocessing=False, test_size=0.2, hyperparameters = svm_hyperparameters[j])

        startTime = time.time()

        svmModel.fit(df)
        svmTime = time.time() - startTime
        
        svmModel.evaluate(dataset_name, encoderNames[j], svmTime, 0)



# runAutoGluonTests()
# run_svm_grid()
run_svm_model()

