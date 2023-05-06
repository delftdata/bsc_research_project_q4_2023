from sklearn import preprocessing
import category_encoders
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
    train_columns = df.drop([target_column], axis = 1).columns
    df[train_columns] = normalizer.fit_transform(df[train_columns])
    return df

class OneHotEncoder:
    def __init__(self):
        self.encoder = preprocessing.OneHotEncoder()

    def encode(self, df, target_column):
        categoricalColumns = getCategoricalColumns(df.drop([target_column], axis = 1))
        self.encoder.fit(df[categoricalColumns])
        onehot = self.encoder.transform(df[categoricalColumns]).toarray()

        onehot_df = pd.DataFrame(onehot, columns=self.encoder.get_feature_names_out(categoricalColumns))
        df = df.drop(categoricalColumns, axis=1)
        df = pd.concat([df, onehot_df], axis=1)

        df = normalizeColumns(df, target_column)
        
        return df

class OrdinalEncoder:
    def __init__(self):
        self.encoder = preprocessing.OrdinalEncoder()
        
    def encode(self, df, target_column):
        categoricalColumns = getCategoricalColumns(df.drop([target_column], axis = 1))
        self.encoder.fit(df[categoricalColumns])
        ordinal = self.encoder.transform(df[categoricalColumns])

        ordinal_df = pd.DataFrame(ordinal, columns=categoricalColumns)
        df = df.drop(categoricalColumns, axis=1)
        df = pd.concat([df, ordinal_df], axis=1)

        df = normalizeColumns(df, target_column)
        
        return df
    
class TargetEncoder:
    def __init__(self):
        self.encoder = category_encoders.TargetEncoder()
        
    def encode(self, df, target_column):
        categoricalColumns = getCategoricalColumns(df.drop([target_column], axis = 1))
        
        target = df[[target_column]]
        df = df.drop(target_column, axis = 1)
        non_categorical_df = df.drop(categoricalColumns, axis = 1)
        df = df.drop(non_categorical_df, axis = 1)

        self.encoder.fit(df, target)
        df = self.encoder.transform(df)

        df = pd.concat([df, non_categorical_df, target], axis = 1)
        df = normalizeColumns(df, target_column)
        
        return df
    
class CatBoostEncoder:
    def __init__(self):
        self.encoder = category_encoders.CatBoostEncoder()
        
    def encode(self, df, target_column):
        categoricalColumns = getCategoricalColumns(df.drop([target_column], axis = 1))
        
        target = df[[target_column]]
        df = df.drop(target_column, axis = 1)
        non_categorical_df = df.drop(categoricalColumns, axis = 1)
        df = df.drop(non_categorical_df, axis = 1)

        self.encoder.fit(df, target)
        df = self.encoder.transform(df)

        df = pd.concat([df, non_categorical_df, target], axis = 1)
        df = normalizeColumns(df, target_column)
        
        return df

class CountEncoder:
    def __init__(self):
        self.encoder = category_encoders.CountEncoder()
        
    def encode(self, df, target_column):
        categoricalColumns = getCategoricalColumns(df.drop([target_column], axis = 1))
        
        target = df[[target_column]]
        df = df.drop(target_column, axis = 1)
        non_categorical_df = df.drop(categoricalColumns, axis = 1)
        df = df.drop(non_categorical_df, axis = 1)

        self.encoder.fit(df, target)
        df = self.encoder.transform(df)

        df = pd.concat([df, non_categorical_df, target], axis = 1)
        df = normalizeColumns(df, target_column)
        
        return df