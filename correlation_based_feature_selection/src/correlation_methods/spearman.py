"""
Module for feature selection using Spearman method.
"""
from scipy.stats import spearmanr


class SpearmanFeatureSelection:
    """
    Class to perform feature selection using the Spearman method.

    Methods
    -------
    compute_correlation(feature, target_feature): Computes the value of the correlation
    """
    @staticmethod
    def compute_correlation(feature, target_feature):
        """
        Calculates the correlation between the feature and target feature using the Spearman method.
        It is similar to Pearson method, but it transforms the features using fractional ranking. It
        can take values between -1 and 1. A value of 0 means that the features are independent. A value
        closer to -1 means that the features are negatively correlated, whereas a value closer to
        1 means that the features are positively correlated. It is a measure of monotonic correlation
        between two features.

        Parameters
        ----------
        feature (DataFrame column): Feature in the data set
        target_feature (DataFrame column): Target feature of the data set

        Returns
        -------
        spearman (float): Correlation between the two features measured using
                          Spearman method
        """
        spearman = spearmanr(feature, target_feature).statistic

        return spearman

    @staticmethod
    def feature_selection(dataframe, target_feature):
        # TODO: add the functionality
        return dataframe, target_feature
