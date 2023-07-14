
from run_combined_test import run_combined_test
import encoders.encoders as encoders
from encoders.encoders import getCategoricalColumns
import pandas as pd
from models.models import AutogluonModel, SVMModel
from models.models import writeToFile
import multiprocessing
n_jobs = multiprocessing.cpu_count()-1
import time



# # Bank Marketing
# file = '../datasets/BankMarketing/bank.csv'
# dataset_name = 'BankMarketing'
# df = pd.read_csv(file)
# label = 'y'
# problemType = 'binary'
# df[label] = df[label].replace('yes', 1)
# df[label] = df[label].replace('no', 0)
# df[label] = df[label].astype('int')
# run_combined_test(df, dataset_name, label, problemType)


# #Census Income
# file = '../datasets/CensusIncome/CensusIncome.csv'
# df = pd.read_csv(file)
# dataset_name = 'CensusIncome'
# label = 'income_label'
# problemType = 'binary'
# run_combined_test(df, dataset_name, label, problemType)


# # Nasa Numeric
# file = '../datasets/NasaNumeric/nasa_numeric.csv'
# dataset_name = 'NasaNumeric'
# df = pd.read_csv(file)
# df = df.drop(['recordnumber'], axis = 1)
# label = 'act_effort'
# problemType = 'regression'
# df = df.fillna(0)
# run_combined_test(df, dataset_name, label, problemType)


# #Housing Prices
# file = '../datasets/housing-prices/housing_prices.csv'
# dataset_name = 'HousingPrices'
# df = pd.read_csv(file)
# label = 'SalePrice'
# problemType = 'regression'
# run_combined_test(df, dataset_name, label, problemType)

# # Connect-4
# file = '../datasets/Connect4/connect4.csv'
# dataset_name = 'Connect4'
# df = pd.read_csv(file)
# label = 'label'
# problemType = 'multiclass'

# df[label] = df[label].replace('win', 2)
# df[label] = df[label].replace('draw', 1)
# df[label] = df[label].replace('loss', 0)
# df[label] = df[label].astype('int')
# run_combined_test(df, dataset_name, label, problemType)

# Nursery
file = '../datasets/Nursery/nursery.csv'
dataset_name = 'Nursery'
df = pd.read_csv(file)
label = 'label'
problemType = 'multiclass'
df = df.fillna(0)
df[label] = df[label].replace('not_recom', 0)
df[label] = df[label].replace('spec_prior', 1)
df[label] = df[label].replace('recommend', 2)
df[label] = df[label].replace('very_recom', 3)
df[label] = df[label].replace('priority', 4)
df[label] = df[label].astype('int')
run_combined_test(df, dataset_name, label, problemType)