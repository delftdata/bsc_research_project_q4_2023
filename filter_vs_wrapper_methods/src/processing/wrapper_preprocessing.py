import pandas as pd
from processing.preprocessing import (discretize_columns_ordinal_encoder,
                                      drop_constant_feature, normalize_df)


def preprocess_sequential_feature_selection(df: pd.DataFrame, excluded_columns: list[str]) -> pd.DataFrame:
    df = discretize_columns_ordinal_encoder(df, excluded_columns)
    df = normalize_df(df, excluded_columns)
    df = drop_constant_feature(df, excluded_columns=excluded_columns)
    return df
