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
    def feature_selection(train_dataframe, target_feature, number_features):
        """
        SELECT K BEST ALGORITHM: Performs feature selection using the Pearson correlation-based method. Selects
        a specified number of top-performing features.

        Parameters
        ----------
        train_dataframe (DataFrame): Training data containing the features
        target_feature (str): Name of the target feature column
        number_features (int): Number of best-performing features to select

        Returns
        -------
        selected_features (list): List of selected features based on the Pearson correlation using "Select k best"
        """
        target_column = train_dataframe[target_feature]
        train_dataframe = train_dataframe.drop(columns=[target_feature])

        # Calculate the Pearson correlation between each feature and the target feature
        pearson_correlations = train_dataframe \
            .apply(func=lambda feature: PearsonFeatureSelection.compute_correlation(feature, target_column),
                   axis=0)

        # Select the top features with the highest absolute correlation
        sorted_correlations = pearson_correlations.abs().sort_values(ascending=False)
        return sorted_correlations[:number_features].index.tolist()

    @staticmethod
    def feature_selection_second_approach(train_dataframe, target_feature, threshold):
        """
        SELECT ABOVE K ALGORITHM: Performs feature selection using the Pearson correlation-based method. Selects
        a number of features that have correlation with the target above a certain threshold.

        Parameters
        ----------
        train_dataframe (DataFrame): Training data containing the features
        target_feature (str): Name of the target feature column
        threshold (float): Minimum value for the feature to be considered useful for predicting the target

        Returns
        -------
        selected_features (list): List of selected features based on the Pearson correlation using "Select above k"
        """
        target_column = train_dataframe[target_feature]
        train_dataframe = train_dataframe.drop(columns=[target_feature])

        # Calculate the Pearson correlation between each feature and the target feature
        pearson_correlations = train_dataframe \
            .apply(func=lambda feature: PearsonFeatureSelection.compute_correlation(feature, target_column),
                   axis=0)

        # Select the top features with the highest absolute correlation
        filtered_features = [feature for feature, correlation in pearson_correlations if
                             correlation > threshold]
        return filtered_features
