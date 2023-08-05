"""
Module for feature selection using Symmetrical Uncertainty method.
"""
from .utility.entropy_estimators import calculate_entropy
from .utility.mutual_information import calculate_information_gain


class SymmetricUncertaintyFeatureSelection:
    """
    Class to perform feature selection using the information-theory based Symmetrical Uncertainty method.

    Methods
    -------
    compute_correlation(feature, target_feature): Computes the value of the correlation using Symmetrical Uncertainty
    feature_selection(train_dataframe, target_feature): Ranks all the features according to their Symmetrical
    Uncertainty value computed with the target
    """
    @staticmethod
    def compute_correlation(feature, target_feature):
        """
        Calculates the correlation between the feature and target feature using the Symmetrical Uncertainty method.
        A value of 0 means that the features are independent, whereas a value of 1 means that knowledge of the feature’s
        value strongly represents target’s value. Source of the code is: https://github.com/jundongl/scikit-feature.

        Parameters
        ----------
        feature (DataFrame column): Feature in the dataset
        target_feature (DataFrame column): Target feature of the dataset

        Returns
        -------
        symmetric_uncertainty (float): Correlation between the two features measured using
        the Symmetrical Uncertainty method
        """
        # Calculate information gain of feature and target
        gain = calculate_information_gain(feature, target_feature)
        # Calculate entropy of feature
        feature_entropy = calculate_entropy(feature)
        # Calculate entropy of target feature
        target_feature_entropy = calculate_entropy(target_feature)
        # Calculate the symmetrical uncertainty between feature and target feature
        symmetric_uncertainty = 2.0 * gain / (feature_entropy + target_feature_entropy)

        return symmetric_uncertainty

    @staticmethod
    def feature_selection(train_dataframe, target_feature):
        """
        Performs feature selection using the Symmetrical Uncertainty method. Ranks all the features. After calling this
        method, a specified number k of top-performing features can be selected.

        Parameters
        ----------
        train_dataframe (DataFrame): Training data containing the features
        target_feature (str): Name of the target feature

        Returns
        -------
        selected_features (list): List of ranked features using the Symmetrical Uncertainty method
        """
        target_column = train_dataframe[target_feature]
        train_dataframe = train_dataframe.drop(columns=[target_feature])

        # Calculate the Symmetrical Uncertainty value between each feature and the target feature
        su_correlations = train_dataframe\
            .apply(func=lambda feature: SymmetricUncertaintyFeatureSelection.compute_correlation(feature, target_column),
                   axis=0)

        # Rank the features in order of the Symmetrical Uncertainty value
        sorted_correlations = su_correlations.sort_values(ascending=False)

        return sorted_correlations.index.tolist(), sorted_correlations.values.tolist()
