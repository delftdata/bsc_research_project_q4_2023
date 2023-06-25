import encoders.encoders as encoders
from encoders.encoders import getCategoricalColumns
import pandas as pd
from models.models import AutogluonModel, SVMModel
from models.models import writeToFile
import multiprocessing
n_jobs = multiprocessing.cpu_count()-1
import time

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



file = '../datasets/housing-prices/train.csv'
dataset_name = 'HousingPrices'
df = pd.read_csv(file)

label = 'SalePrice'
problemType = 'regression'

encoderNames = [ 'onehot', 'ordinal', 'target', 'catboost', 'count']

encoders_list = [encoders.OneHotEncoder(), encoders.OrdinalEncoder(
    ), encoders.TargetEncoder(), encoders.CatBoostEncoder(), encoders.CountEncoder()]

algorithms = [
    # {
    #     'LR': {}
    # },
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
        # 'LR',
          'XGB',
        'GBM', 'RF'
        ]


df = df.head(1000)
df_copy = df
categorical_columns = getCategoricalColumns(df)
categorical_columns = ['Neighborhood', 'Exterior2nd', 'BsmtFinType1']
for k, column in enumerate(categorical_columns):
    print(column + ': ' + str(df[column].unique().size))


for k, column in enumerate(categorical_columns):
    #try encoding this column differently
    print(column)
    df = df_copy
    this_column = pd.concat([df[column], df[label]], axis= 1)
    other_columns = df.drop(column, axis = 1)

    other_encoded = encoders.TargetEncoder().encode(other_columns, label)
    for i, algorithm in enumerate(algorithms):
        max_score = 10000000000
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
            if(abs(score) < max_score):
                max_score = abs(score)
                encoderName = encoderNames[j]
        # for column k, algorithm i => encoder j is the best
        content = f"For column {column} and algorithm {algorithmNames[i]} => encoder {encoderName} is the best: {max_score}"
        writeToFile('results_combined/'  + dataset_name + '/results' , content)
# runAutoGluonTests()
# run_svm_model()


