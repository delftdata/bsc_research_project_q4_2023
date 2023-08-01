"""
Module for feature selection using Information Gain method.
"""
from .utility.mutual_information import calculate_information_gain


class InformationGainFeatureSelection:
    """
    Class to perform feature selection using the Information Gain method.

    Methods
    -------
    compute_correlation(feature, target_feature): Computes the value of the correlation
    feature_selection(train_dataframe, target_feature): Ranks the features according to their relevance for predicting
    the target
    """
    @staticmethod
    def compute_correlation(feature, target_feature):
        """
        SELECT K BEST ALGORITHM: Calculates the correlation between the feature and target feature using the Information
        Gain method. A value of 0 means that the features are independent, whereas a value
        of 1 means that knowledge of the feature’s value strongly represents target’s value. Source
        of the code is: https://github.com/jundongl/scikit-feature.

        Parameters
        ----------
        feature (DataFrame column): Feature in the dataset
        target_feature (DataFrame column): Target feature of the dataset

        Returns
        -------
        information_gain (float): Correlation between the two features measured using Information Gain
        """
        return calculate_information_gain(feature, target_feature)

    @staticmethod
    def feature_selection(train_dataframe, target_feature):
        """
        Performs feature selection using the Information Gain correlation-based method. Selects
        a specified number of top-performing features.

        Parameters
        ----------
        train_dataframe (DataFrame): Training data containing the features
        target_feature (str): Name of the target feature

        Returns
        -------
        selected_features (list): List of selected features using Information Gain
        """
        target_column = train_dataframe[target_feature]
        train_dataframe = train_dataframe.drop(columns=[target_feature])

        # Calculate the Symmetric Uncertainty correlation between each feature and the target feature
        ig_correlations = train_dataframe\
            .apply(func=lambda feature: InformationGainFeatureSelection.compute_correlation(feature, target_column),
                   axis=0)

        # Select the top features with the highest correlation
        sorted_correlations = ig_correlations.sort_values(ascending=False)

        return sorted_correlations.index.tolist(), sorted_correlations.values.tolist()