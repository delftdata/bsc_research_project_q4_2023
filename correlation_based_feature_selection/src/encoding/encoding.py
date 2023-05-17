from sklearn import preprocessing
from pandas.api.types import is_object_dtype
import pandas as pd

def get_categorical_features(dataframe):
    for column in dataframe.dtypes.items():
        if is_object_dtype(column):
            dataframe[column] = dataframe[column].apply(lambda x: x.decode("utf-8"))

    aux_dataframe = dataframe.astype(dataframe.infer_objects().dtypes)

    return aux_dataframe.select_dtypes(include=['object']).columns


def normalize_features(dataframe, target_feature):
    features = dataframe.drop([target_feature], axis=1).columns

    normalizer = preprocessing.Normalizer()
    dataframe[features] = normalizer.fit_transform(dataframe[features])

    return dataframe

class OneHotEncoder:
    def __init__(self):
        self.encoder = preprocessing.OneHotEncoder()

    def encode(self, dataframe, target_feature):
        categorical_features = get_categorical_features(dataframe.drop([target_feature], axis=1))

        self.encoder.fit(dataframe[categorical_features])
        onehot = self.encoder.transform(dataframe[categorical_features]).toarray()

        onehot_dataframe = pd.DataFrame(
            onehot, columns=self.encoder.get_feature_names_out(categorical_features))

        dataframe = dataframe.drop(categorical_features, axis=1)
        dataframe = pd.concat([dataframe, onehot_dataframe], axis=1)
        dataframe = normalize_features(dataframe, target_feature)

        return dataframe