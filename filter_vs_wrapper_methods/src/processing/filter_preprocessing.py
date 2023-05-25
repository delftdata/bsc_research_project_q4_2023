import pandas as pd
from processing.preprocessing import Preprocessing


def preprocess_chi2(df: pd.DataFrame, excluded_columns: list[str]) -> pd.DataFrame:
    df = Preprocessing.discretize_columns_ordinal_encoder(df, excluded_columns)
    df = Preprocessing.scale_columns_min_max(df, excluded_columns)
    df = Preprocessing.discretize_columns_k_bins(df, excluded_columns)
    df = Preprocessing.drop_constant_feature(df, excluded_columns)
    return df


def preprocess_anova(df: pd.DataFrame, excluded_columns: list[str]) -> pd.DataFrame:
    df = Preprocessing.discretize_columns_ordinal_encoder(df, excluded_columns)
    df = Preprocessing.normalize_df(df, excluded_columns)
    df = Preprocessing.drop_constant_feature(df, excluded_columns=excluded_columns)
    return df
