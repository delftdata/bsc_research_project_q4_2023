import encoders.encoders as encoders
from encoders.encoders import getCategoricalColumns
import pandas as pd
from models.models import AutogluonModel, SVMModel
from models.models import writeToFile
import multiprocessing
n_jobs = multiprocessing.cpu_count()-1
import time


    

def runAutoGluonTests():
   
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

def run_svm_model():
    for j, encoder in enumerate(encoders_list):
        print(f'Running encoder {encoderNames[j]}')
        svmModel = SVMModel(problem_type=problemType, label=label,
                        data_preprocessing=False, test_size=0.2, hyperparameters = svm_hyperparameters[j])

        startTime = time.time()
        encoded_df = encoder.encode(df, label)
        encodeTime = time.time() - startTime

        svmModel.fit(encoded_df)
        svmTime = time.time() - startTime
        
        svmModel.evaluate(dataset_name, encoderNames[j], svmTime, encodeTime)



file = '../datasets/BankMarketing/bank.csv'
dataset_name = 'BankMarketing'
df = pd.read_csv(file)
label = 'y'
problemType = 'binary'
df[label] = df[label].replace('yes', 1)
df[label] = df[label].replace('no', 0)

encoderNames = [ 'onehot', 'ordinal', 'target', 'catboost', 'count']

encoders_list = [encoders.OneHotEncoder(), encoders.OrdinalEncoder(
    ), encoders.TargetEncoder(), encoders.CatBoostEncoder(), encoders.CountEncoder()]

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
    }
    ]

algorithmNames = [
        'LR', 'XGB',
        'GBM', 'RF'
        ]


df = df.head(1000)
df_copy = df
categorical_columns = getCategoricalColumns(df)

for k, column in enumerate(categorical_columns.drop([])):
    #try encoding this column differently
    print(column)
    df = df_copy
    this_column = pd.concat([df[column], df[label]], axis= 1)
    other_columns = df.drop(column, axis = 1)

    other_encoded = encoders.TargetEncoder().encode(other_columns, label)
    for i, algorithm in enumerate(algorithms):
        max_score = 0
        encoder = ""
        encoderName = "undefined"
        for j, encoder in enumerate(encoders_list):
            this_encoded = encoder.encode(pd.DataFrame(this_column), label)
            df = pd.concat([other_encoded, this_encoded.drop(label, axis = 1)], axis=1)
            

            manualEncodingModel = AutogluonModel(
                problem_type=problemType, label=label, data_preprocessing=False, test_size=0.2, hyperparameters=algorithm)

            manualEncodingModel.fit(df)
            score = manualEncodingModel.evaluate(
                dataset_name,  algorithmNames[i], encoderNames[j], 0, 0, test_type = 'combined')
            if(score > max_score):
                max_score = score
                encoderName = encoderNames[j]
        # for column k, algorithm i => encoder j is the best
        content = f"For column {column} and algorithm {algorithmNames[i]} => encoder {encoderName} is the best: {max_score}"
        writeToFile('results_combined/'  + dataset_name + '/results' , content)
# runAutoGluonTests()
# run_svm_model()


