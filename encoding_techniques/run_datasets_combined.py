
from run_combined_test import run_combined_test
import encoders.encoders as encoders
from encoders.encoders import getCategoricalColumns
import pandas as pd
from models.models import AutogluonModel, SVMModel
from models.models import writeToFile
import multiprocessing
n_jobs = multiprocessing.cpu_count()-1
import time



# Bank Marketing
file = '../datasets/BankMarketing/bank.csv'
dataset_name = 'BankMarketing'
df = pd.read_csv(file)
label = 'y'
problemType = 'binary'
df[label] = df[label].replace('yes', 1)
df[label] = df[label].replace('no', 0)
df[label] = df[label].astype('int')
run_combined_test(df, dataset_name, label, problemType)


# #Census Income
# file = '../datasets/CensusIncome/CensusIncome.csv'
# df = pd.read_csv(file)
# dataset_name = 'CensusIncome'
# label = 'income_label'
# problemType = 'binary'
# run_combined_test(df, dataset_name, label, problemType)



