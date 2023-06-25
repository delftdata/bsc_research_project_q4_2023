import encoders.encoders as encoders
from encoders.encoders import getCategoricalColumns
import pandas as pd
from models.models import AutogluonModel, SVMModel
from models.models import writeToFile
import multiprocessing
n_jobs = multiprocessing.cpu_count()-1
import time
from random import randint

def encode_randomly(df, label):
    number = randint(0, 5)
    if(number == 0): return encoders.OneHotEncoder().encode(df, label)
    elif(number == 1): return encoders.OrdinalEncoder().encode(df, label)
    elif(number == 2): return encoders.TargetEncoder().encode(df, label)
    elif(number == 3): return encoders.CatBoostEncoder().encode(df, label)
    else: return encoders.CountEncoder().encode(df, label)
    
def run_combined_test(df, dataset_name, label, problemType):
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
            'LR',
            'XGB',
            'GBM', 'RF'
            ]

    df_copy = df
    categorical_columns = getCategoricalColumns(df)

    encodedDf = pd.DataFrame(df[label])

    startTime = time.time()
    for k, column in enumerate(categorical_columns):
        this_column = pd.concat([df[column], df[label]], axis= 1)
        uniques = df[column].unique().size
        if uniques <= 4:
            this_encoded = encoders.OneHotEncoder().encode(this_column, label)
        elif uniques <= 15:
            this_encoded = encoders.OrdinalEncoder().encode(this_column, target_column = label)
        else:
            this_encoded = encoders.TargetEncoder().encode(this_column, label) 
        # this_encoded = encode_randomly(this_column, label)
        this_encoded = this_encoded.drop(label, axis = 1)
        
        encodedDf = pd.concat([encodedDf, this_encoded], axis = 1)

    encodeTime = time.time() - startTime
    encodedDf = pd.DataFrame(encodedDf)

    for i, algorithm in enumerate(algorithms):
        startTime = time.time()

        manualEncodingModel = AutogluonModel(
            problem_type=problemType, label=label, data_preprocessing=False, test_size=0.2, hyperparameters=algorithm)

        manualEncodingModel.fit(encodedDf)

        fitTime = time.time() - startTime
        
        manualEncodingModel.evaluate(
            dataset_name,  algorithmNames[i], 'CombinedEncoders', encodeTime, fitTime+encodeTime, test_type = 'combinedFinal')


    startTime = time.time()
    svmModel = SVMModel(problem_type=problemType, label=label,
                            data_preprocessing=False, test_size=0.2)
    svmModel.fit(encodedDf)
    fitTime = time.time() - startTime
    svmModel.evaluate(dataset_name, '', fitTime, encodeTime, test_type = 'combinedFinal')
