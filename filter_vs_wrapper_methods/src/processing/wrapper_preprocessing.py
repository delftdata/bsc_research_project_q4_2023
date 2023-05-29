import pandas as pd
from processing.preprocessing import Preprocessing


def preprocess_sequential_feature_selection(df: pd.DataFrame, excluded_columns: list[str]) -> pd.DataFrame:
    df = Preprocessing.discretize_columns_ordinal_encoder(df, excluded_columns)
    df = Preprocessing.normalize_df(df, excluded_columns)
    df = Preprocessing.drop_constant_feature(df, excluded_columns=excluded_columns)
    return df
