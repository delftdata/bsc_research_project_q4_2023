"""
Module for feature selection using Symmetric Uncertainty method.
"""
from .utility.entropy_estimators import calculate_entropy
from .utility.mutual_information import calculate_information_gain


class SymmetricUncertaintyFeatureSelection:
    """
    Class to perform feature selection using the Symmetric Uncertainty method.

    Methods
    -------
    compute_correlation(feature, target_feature): Computes the value of the correlation
    """
    @staticmethod
    def compute_correlation(feature, target_feature):
        """
        Calculates the correlation between the feature and target feature using the Symmetric
        Uncertainty method. A value of 0 means that the features are independent, whereas a value
        of 1 means that knowledge of the feature’s value strongly represents target’s value.

        Parameters
        ----------
        feature (DataFrame column): Feature in the data set
        target_feature (DataFrame column): Target feature of the data set

        Returns
        -------
        symmetric_uncertainty (float): Correlation between the two features measured using
                                       Symmetric Uncertainty method
        """
        # Calculate information gain of feature and target
        gain = calculate_information_gain(feature, target_feature)
        # Calculate entropy of feature
        feature_entropy = calculate_entropy(feature)
        # Calculate entropy of target feature
        target_feature_entropy = calculate_entropy(target_feature)
        # Calculate the symmetric uncertainty between feature and target feature
        symmetric_uncertainty = 2.0 * gain / (feature_entropy + target_feature_entropy)

        return symmetric_uncertainty

    @staticmethod
    def feature_selection(dataframe, target_feature):
        # TODO: add the functionality
        return dataframe, target_feature
