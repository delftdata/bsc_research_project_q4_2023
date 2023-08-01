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
        SELECT K BEST ALGORITHM: Calculates the correlation between the feature and target feature using the Symmetric
        Uncertainty method. A value of 0 means that the features are independent, whereas a value
        of 1 means that knowledge of the feature’s value strongly represents target’s value. Source
        of the code is: https://github.com/jundongl/scikit-feature.

        Parameters
        ----------
        feature (DataFrame column): Feature in the dataset
        target_feature (DataFrame column): Target feature of the dataset

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
    def feature_selection(train_dataframe, target_feature):
        """
        Performs feature selection using the Symmetric Uncertainty correlation-based method. Selects
        a specified number of top-performing features.

        Parameters
        ----------
        train_dataframe (DataFrame): Training data containing the features
        target_feature (str): Name of the target feature

        Returns
        -------
        selected_features (list): List of selected features using the Symmetric Uncertainty correlation
        """
        target_column = train_dataframe[target_feature]
        train_dataframe = train_dataframe.drop(columns=[target_feature])

        # Calculate the Symmetric Uncertainty correlation between each feature and the target feature
        su_correlations = train_dataframe\
            .apply(func=lambda feature: SymmetricUncertaintyFeatureSelection.compute_correlation(feature, target_column),
                   axis=0)

        # Select the top features with the highest correlation
        sorted_correlations = su_correlations.sort_values(ascending=False)

        return sorted_correlations.index.tolist(), sorted_correlations.values.tolist()
