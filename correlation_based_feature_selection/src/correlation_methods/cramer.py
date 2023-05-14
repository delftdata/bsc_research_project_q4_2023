"""
Module for feature selection using Cramer's V method.
"""
from pandas import crosstab
from scipy.stats import contingency


class CramersVFeatureSelection:
    """
    Class to perform feature selection using Cramer's V method.

    Methods
    -------
    compute_correlation(feature, target_feature): Computes the value of the correlation
    """
    @staticmethod
    def compute_correlation(feature, target_feature):
        """
        Calculates the correlation between the feature and target feature using Cramer's V method.
        It is computed by taking the square root of the chi-squared statistic divided by the number
        of the observations and the minimum dimension of the features minus 1. A value of 0 means
        that the features are not associated, whereas a value of 1 means that the features are
        perfectly associated.

        Parameters
        ----------
        feature (DataFrame column): Feature in the data set
        target_feature (DataFrame column): Target feature of the data set

        Returns
        -------
        cramers_v (float): Correlation between the two features measured using
                           Cramer's V method
        """
        contingency_table_values = crosstab(feature, target_feature).values
        cramers_v = contingency.association(contingency_table_values, method="cramer")

        return cramers_v

    @staticmethod
    def feature_selection(dataframe, target_feature):
        # TODO: add the functionality
        return dataframe, target_feature
