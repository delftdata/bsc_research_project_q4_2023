import pandas as pd

from processing.preprocessing import (discretize_columns_k_bins,
                                      discretize_columns_ordinal_encoder,
                                      drop_constant_feature, normalize_df,
                                      scale_columns_min_max)


def preprocess_chi2(df: pd.DataFrame, excluded_columns: list[str]) -> pd.DataFrame:
    """Preprocesses the DataFrame for Chi-Squared feature selection.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.
    excluded_columns : list[str]
        The list of columns to exclude from preprocessing.

    Returns
    -------
    pd.DataFrame
        The preprocessed DataFrame.
    """
    df = discretize_columns_ordinal_encoder(df, excluded_columns)
    df = scale_columns_min_max(df, excluded_columns)
    df = discretize_columns_k_bins(df, [])
    df = drop_constant_feature(df, excluded_columns)
    return df


def preprocess_anova(df: pd.DataFrame, excluded_columns: list[str], normalization: bool) -> pd.DataFrame:
    """Preprocesses the DataFrame for ANOVA feature selection.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.
    excluded_columns : list[str]
        The list of columns to exclude from preprocessing.
    normalization : bool
        A flag indicating whether to perform normalization.

    Returns
    -------
    pd.DataFrame
        The preprocessed DataFrame.
    """
    df = discretize_columns_ordinal_encoder(df, excluded_columns)
    if normalization:
        df = normalize_df(df, excluded_columns)
    df = drop_constant_feature(df, excluded_columns=excluded_columns)
    return df
