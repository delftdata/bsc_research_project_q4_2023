from sklearn import preprocessing
import pandas as pd
from pandas.api.types import is_object_dtype


def getCategoricalColumns(df):
    for col in df.dtypes.items():
        if is_object_dtype(col):
            df[col] = df[col].apply(lambda x: x.decode("utf-8"))

    df = df.astype(df.infer_objects().dtypes)
    return df.select_dtypes(include=['object']).columns


def normalizeColumns(df, target_column):
    normalizer = preprocessing.Normalizer()
    train_columns = df.drop([target_column], axis=1).columns

    # NA imputation is handled before encoding the categorical variables.  There should be no 'na' values before executing this line of code.
    df = df.fillna(0)
    df[train_columns] = normalizer.fit_transform(df[train_columns])

    return df


def scaleColumns(df, target_column):
    normalizer = preprocessing.MinMaxScaler()
    train_columns = df.drop([target_column], axis=1).columns
    df[train_columns] = normalizer.fit_transform(df[train_columns], df[target_column])

    # print(df.head(1))
    return df


class OneHotEncoder:
    def __init__(self):
        self.encoder = preprocessing.OneHotEncoder()

    def encode(self, df, target_column):
        categoricalColumns = getCategoricalColumns(
            df.drop([target_column], axis=1))

        target_df = df[target_column]

        self.encoder.fit(df[categoricalColumns])
        onehot = self.encoder.transform(df[categoricalColumns]).toarray()

        onehot_df = pd.DataFrame(
            onehot, columns=self.encoder.get_feature_names_out(categoricalColumns))
        df = df.drop(categoricalColumns, axis=1)
        df = pd.concat([df, onehot_df], axis=1)

        df = normalizeColumns(df, target_column)

        # df = pd.concat([df, target_df], axis = 1)

        return df
