"""
Module for feature selection using Pearson method.
"""
from scipy.stats import pearsonr


class PearsonFeatureSelection:
    """
    Class to perform feature selection using the Pearson method.

    Methods
    -------
    compute_correlation(feature, target_feature): Computes the value of the correlation
    """
    @staticmethod
    def compute_correlation(feature, target_feature):
        """
        Calculates the correlation between the feature and target feature using the Pearson method.
        It can take values between -1 and 1. A value of 0 means that the features are independent. A value
        closer to -1 means that the features are negatively correlated, whereas a value closer to
        1 means that the features are positively correlated. It is a measure of linear correlation
        between two features.

        Parameters
        ----------
        feature (DataFrame column): Feature in the data set
        target_feature (DataFrame column): Target feature of the data set

        Returns
        -------
        pearson (float): Correlation between the two features measured using
                         Pearson method
        """
        pearson = pearsonr(feature, target_feature).statistic

        return pearson

    @staticmethod
    def feature_selection(dataframe, target_feature):
        # TODO: add the functionality
        return dataframe, target_feature
