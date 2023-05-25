from sklearn import preprocessing
import category_encoders
import pandas as pd
from pandas.api.types import is_object_dtype
from autogluon.features.generators import AutoMLPipelineFeatureGenerator


def getCategoricalColumns(df):
    for col in df.dtypes.items():
        if is_object_dtype(col):
            df[col] = df[col].apply(lambda x: x.decode("utf-8"))

    df = df.astype(df.infer_objects().dtypes)
    return df.select_dtypes(include=['object']).columns


def normalizeColumns(df, target_column):
    normalizer = preprocessing.Normalizer()
    train_columns = df.drop([target_column], axis=1).columns
    
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


class OrdinalEncoder:
    def __init__(self):
        self.encoder = preprocessing.OrdinalEncoder()

    def encode(self, df, target_column):
        categoricalColumns = getCategoricalColumns(
            df.drop([target_column], axis=1))
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
        categoricalColumns = getCategoricalColumns(
            df.drop([target_column], axis=1))

        target = df[[target_column]]
        df = df.drop(target_column, axis=1)
        non_categorical_df = df.drop(categoricalColumns, axis=1)
        df = df.drop(non_categorical_df, axis=1)
        # print(categoricalColumns)
        # print(df.head(5))
        # print(target.head(5))
        self.encoder.fit(df, target)
        df = self.encoder.transform(df)

        df = pd.concat([df, non_categorical_df, target], axis=1)
        df = normalizeColumns(df, target_column)

        return df


class CatBoostEncoder:
    def __init__(self):
        self.encoder = category_encoders.CatBoostEncoder()

    def encode(self, df, target_column):
        categoricalColumns = getCategoricalColumns(
            df.drop([target_column], axis=1))

        target = df[[target_column]]
        df = df.drop(target_column, axis=1)
        non_categorical_df = df.drop(categoricalColumns, axis=1)
        df = df.drop(non_categorical_df, axis=1)

        self.encoder.fit(df, target)
        df = self.encoder.transform(df)

        df = pd.concat([df, non_categorical_df, target], axis=1)
        df = normalizeColumns(df, target_column)

        return df


class CountEncoder:
    def __init__(self):
        self.encoder = category_encoders.CountEncoder()

    def encode(self, df, target_column):
        categoricalColumns = getCategoricalColumns(
            df.drop([target_column], axis=1))

        target = df[[target_column]]
        df = df.drop(target_column, axis=1)
        non_categorical_df = df.drop(categoricalColumns, axis=1)
        df = df.drop(non_categorical_df, axis=1)

        self.encoder.fit(df, target)
        df = self.encoder.transform(df)

        df = pd.concat([df, non_categorical_df, target], axis=1)
        df = normalizeColumns(df, target_column)

        return df
    
class AutoGluonEncoder:
    def __init__(self):
        self.encoder = AutoMLPipelineFeatureGenerator()

    def encode(self, df, target_column):
        X_train = df.drop(labels=[target_column], axis=1)
        y_train = df[target_column]
        
        df = self.encoder.fit_transform(X=X_train, y=y_train)

        df = df = pd.concat([df, y_train], axis=1)
        df = normalizeColumns(df, target_column)        

        return df


class CombinedEncoder:
    def __init__(self):
        self.encoder = category_encoders.CountEncoder()

    def encode(self, df, target_column):
        categoricalColumns = getCategoricalColumns(
            df.drop([target_column], axis=1))
        
        categorical_df = df[categoricalColumns]
        numeric_df = df.drop(categorical_df, axis = 1)
        numeric_df = numeric_df.drop(target_column, axis = 1)
        numericColumns = numeric_df.columns

        df_onehot = OneHotEncoder().encode(df=df, target_column=target_column)
        df_onehot = df_onehot.drop([target_column], axis = 1)
        df_onehot = df_onehot.drop(numericColumns, axis = 1)
        df_onehot = df_onehot.add_suffix('_onehot')

        df_ordinal = OrdinalEncoder().encode(df=df, target_column=target_column)
        df_ordinal = df_ordinal.drop([target_column], axis = 1)
        df_ordinal = df_ordinal.drop(numericColumns, axis = 1)
        df_ordinal = df_ordinal.add_suffix('_ordinal')
        
        df_target = TargetEncoder().encode(df=df, target_column=target_column)
        df_target = df_target.drop([target_column], axis = 1)
        df_target = df_target.drop(numericColumns, axis = 1)
        df_target = df_target.add_suffix('_target')
        
        df_catboost = CatBoostEncoder().encode(df=df, target_column=target_column)
        df_catboost = df_catboost.drop([target_column], axis = 1)
        df_catboost = df_catboost.drop(numericColumns, axis = 1)
        df_catboost = df_catboost.add_suffix('catboost')
        
        df_count = CountEncoder().encode(df=df, target_column=target_column)
        df_count = df_count.drop([target_column], axis = 1)
        df_count = df_count.drop(numericColumns, axis = 1)
        df_count = df_count.add_suffix('_count')
        
        # print(df_ordinal.columns)
        df = pd.concat([df_onehot, df_ordinal, df_target, df_catboost, df_count, df[numericColumns], df[target_column]], axis = 1)
        return df
