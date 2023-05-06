from models.models import LinearModel, TreeModel, SVMModel
import encoders.encoders as encoders
import pandas as pd

file = 'datasets/CensusIncome.csv'
df = pd.read_csv(file)
label = 'income_label'

# linearModel = LinearModel(problem_type='binary', label=label, data_preprocessing=False, test_size=0.2)
# linearModel.fit(OneHotEncoder().encode(df.head(10000), label))
# print(linearModel.evaluate())

treeModel = TreeModel(problem_type='binary', label=label, data_preprocessing=False, test_size=0.2)
treeModel.fit(encoders.CountEncoder().encode(df.head(10000), label))
print(treeModel.evaluate())

# encoder = OneHotEncoder()
# svmModel = SVMModel(problem_type='binary', label=label, data_preprocessing=False, test_size=0.2)
# svmModel.fit(OneHotEncoder().encode(df.head(10000), label))
# print(svmModel.evaluate())