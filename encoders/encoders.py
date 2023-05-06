from sklearn import preprocessing
import pandas as pd
import numpy as np
from pandas.api.types import is_object_dtype

def getCategoricalColumns(df):
    for col in df.dtypes.items():
        if is_object_dtype(col):
            df[col] = df[col].apply(lambda x: x.decode("utf-8"))
    
    df = df.astype(df.infer_objects().dtypes)
    return df.select_dtypes(include=['object']).columns

class OneHotEncoder:
    def __init__(self):
        self.encoder = preprocessing.OneHotEncoder()
        self.normalizer = preprocessing.Normalizer()
        

    def encode(self, df, target_column):
        categoricalColumns = getCategoricalColumns(df.drop([target_column], axis = 1))
        self.encoder.fit(df[categoricalColumns])
        onehot = self.encoder.transform(df[categoricalColumns]).toarray()

        onehot_df = pd.DataFrame(onehot, columns=self.encoder.get_feature_names_out(categoricalColumns))
        df = df.drop(categoricalColumns, axis=1)
        df = pd.concat([df, onehot_df], axis=1)

        train_columns = df.drop([target_column], axis = 1).columns
        df[train_columns] = self.normalizer.fit_transform(df[train_columns])
        
        return df

# def encode_ordinal(df, categorical_columns):
#     # Create an instance of the Ordinal Encoder
#     encoder = OrdinalEncoder()
#     normalizer = Normalizer()

#     # Fit the encoder on the categorical columns of the DataFrame
#     encoder.fit(df[categorical_columns])

#     # Transform the categorical columns into ordinal encoded columns
#     ordinal = encoder.transform(df[categorical_columns])

#     # Create a new DataFrame with the ordinal encoded columns
#     ordinal_df = pd.DataFrame(ordinal, columns=categorical_columns)

#     # Drop the original categorical variables from the DataFrame
#     df = df.drop(categorical_columns, axis=1)

#     # Concatenate the ordinal encoded DataFrame with the original DataFrame
#     df = pd.concat([df, ordinal_df], axis=1)
    
#     #Normalize data using sk-learn normalizer
#     df[df.columns] = normalizer.fit_transform(df[df.columns])

#     return df

# def encode_catboost(df, categorical_columns, target_column):
#     # Create an instance of the CatBoost Encoder
#     encoder = CatBoostEncoder()

#     target = df[[target_column]]
#     df = df.drop(target_column, axis = 1)
#     non_categorical_df = df.drop(categorical_columns, axis = 1)
#     df = df.drop(non_categorical_df, axis = 1)

#     # Fit encoder and transform the features
#     encoder.fit(df, target)
#     df = encoder.transform(df)

#     df = pd.concat([df, non_categorical_df], axis = 1)
#     scaler = StandardScaler()
#     df[df.columns] = scaler.fit_transform(df[df.columns])

#     df = pd.concat([df, target], axis=1)
    
#     return df



# def encode_target(df, categorical_columns, target_column):
#     # Create an instance of the TargetEncoder
#     encoder = TargetEncoder()

#     target = df[[target_column]]
#     df = df.drop(target_column, axis = 1)
#     non_categorical_df = df.drop(categorical_columns, axis = 1)
#     df = df.drop(non_categorical_df, axis = 1)

#     # Fit encoder and transform the features
#     encoder.fit(df, target)
#     df = encoder.transform(df)

#     df = pd.concat([df, non_categorical_df], axis = 1)
#     scaler = StandardScaler()
#     df[df.columns] = scaler.fit_transform(df[df.columns])

#     df = pd.concat([df, target], axis=1)
    
#     return df

# def encode_count(df, categorical_columns, target_column):
#     # Create an instance of the CountEncoder
#     encoder = CountEncoder()

#     target = df[[target_column]]
#     df = df.drop(target_column, axis = 1)
#     non_categorical_df = df.drop(categorical_columns, axis = 1)
#     df = df.drop(non_categorical_df, axis = 1)

#     # Fit encoder and transform the features
#     encoder.fit(df, target)
#     df = encoder.transform(df)

#     df = pd.concat([df, non_categorical_df], axis = 1)
#     scaler = StandardScaler()
#     df[df.columns] = scaler.fit_transform(df[df.columns])

#     df = pd.concat([df, target], axis=1)
    
#     return df