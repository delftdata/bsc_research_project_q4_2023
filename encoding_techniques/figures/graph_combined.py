import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pandas.api.types import is_object_dtype


def getCategoricalColumns(df):
    for col in df.dtypes.items():
        if is_object_dtype(col):
            df[col] = df[col].apply(lambda x: x.decode("utf-8"))

    df = df.astype(df.infer_objects().dtypes)
    return df.select_dtypes(include=['object']).columns


# autogluon = np.array([
#     [84.55, 87.55, 87.62, 86.11, 79.88], 
#     [88.99, 90.27, 90.12, 90.24, 88.39],
#     [78.01, 98.96, 98.88, 97.49, 89.0],
#     [65.97, 86.55, 86.43, 82.9, 83.05]         
#     ])

# onehot = np.array([
#     [84.81, 85.78, 85.91, 84.24, 79.88],
#     [89.77, 90.77, 90.51, 90.41, 88.39],
#     [92.4, 99.96, 100, 99.61, 100],
#     [77.51, 86.89, 87.06, 84.51, 88.85]
# ])

no_features = ['0-2', '3', '4', '5-10', '10-15', '15+']




algorithm_names = ['Linear model', 'XGBoost', 'LightGBM', 'Random Forest', 'SVM']


def calculate_feature_distribution(file, dataset_name):
    df = pd.read_csv(file)
    categorical_columns = getCategoricalColumns(df)

    graph_values = dict()

    for k, column in enumerate(categorical_columns):
        unique_feature_values = df[column].unique().size
        graph_values[column] = unique_feature_values

    columns = []
    values = []
    
    for column in graph_values.keys():
        # print("here")
        columns.append(column)
        values.append(graph_values[column])
  
    values.sort()


    for i, a in enumerate(columns):
        columns[i] = columns[i][0:2]
    
    plt.clf()
    plt.scatter(columns, values)
    plt.xlabel("feature")
    plt.ylabel("number of unique values")
    # plt.tight_layout()
    # plt.show()
    plt.title(dataset_name)
    plt.savefig(f'dark/feature-distribution/{dataset_name}.png')

def combined_encoders_results(results):
    
    # for i, algorithm_name in enumerate(algorithm_names):
    #     plt.errorbar(algorithm_name, column_means[i], column_std[i],
    #                  linestyle='None', marker='^', capsize=3)
    
    # plt.errorbar('mean of algs', mean, std,
    #                  linestyle='None', marker='^', capsize=3)
    my_dict = {
        'Linear': results.T[0],
        'XGB': results.T[1],
        'GBM': results.T[2],
        'RF': results.T[3],
        'SVM': results.T[4],
    }
    plt.boxplot(my_dict.values(), labels=my_dict.keys())
    plt.xlabel('model')
    plt.ylabel('accuracy')

    # plt.title(f"Accuracy Mean - combining encoders")
    # plt.show()
    plt.savefig(f'dark/accuracy/boxplot-accuracy-combined.png')
    plt.clf()

plt.style.use('ggplot')


dataset_files = ['../../datasets/housing-prices/train.csv',
    '../../datasets/BankMarketing/bank.csv',
    '../../datasets/CensusIncome/CensusIncome.csv',
    '../../datasets/Connect4/connect4.csv',
    '../../datasets/NasaNumeric/nasa_numeric.csv',
    '../../datasets/Nursery/nursery.csv'
]

dataset_names = ['Housing Prices',
    'Bank Marketing',
    'Census Income',
    'Connect-4',
    'Nasa Numeric',
    'Nursery'
]

# for i, file in enumerate(dataset_files):
#     calculate_feature_distribution(file, dataset_names[i])


results =  np.array([
    [76.63, 76.63, 76.63, 76.63, 76.63],
    [82.83, 85.49, 84.8, 92.94, 82.53],
    [89.34, 89.4, 89.26, 89.27, 89.43],
    [75.75, 86.89, 87.01, 84.51, 81.62]
])

column_means = np.mean(results, axis=0)
column_std = np.std(results, axis = 0)
print(column_means)
mean = np.mean(results)
std = np.std(results)
combined_encoders_results(results)