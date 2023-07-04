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
        SELECT K BEST ALGORITHM: Calculates the correlation between the feature and target feature using the Spearman method.
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
    def feature_selection(train_dataframe, target_feature):
        """
        Performs feature selection using the Spearman correlation-based method. Selects
        a specified number of top-performing features.

        Parameters
        ----------
        train_dataframe (DataFrame): Training data containing the features
        target_feature (str): Name of the target feature column

        Returns
        -------
        selected_features (list): List of selected features based on the Spearman correlation
        """
        target_column = train_dataframe[target_feature]
        train_dataframe = train_dataframe.drop(columns=[target_feature])

        # Calculate the Spearman correlation between each feature and the target feature
        pearson_correlations = train_dataframe \
            .apply(func=lambda feature: SpearmanFeatureSelection.compute_correlation(feature, target_column),
                   axis=0)

        # Select the top features with the highest absolute correlation
        sorted_correlations = pearson_correlations.abs().sort_values(ascending=False)

        return sorted_correlations.index.tolist(), sorted_correlations.values.tolist()

    @staticmethod
    def feature_selection_second_approach(train_dataframe, target_feature, threshold):
        """
        SELECT ABOVE C ALGORITHM: Performs feature selection using the Spearman correlation-based method. Selects
        a number of features that have correlation with the target above a certain threshold.

        Parameters
        ----------
        train_dataframe (DataFrame): Training data containing the features
        target_feature (str): Name of the target feature column
        threshold (float): Minimum value for the feature to be considered useful for predicting the target

        Returns
        -------
        selected_features (list): List of selected features based on the Spearman correlation using "Select above c"
        """
        target_column = train_dataframe[target_feature]
        train_dataframe = train_dataframe.drop(columns=[target_feature])

        # Calculate the Spearman correlation between each feature and the target feature
        spearman_correlations = train_dataframe \
            .apply(func=lambda feature: SpearmanFeatureSelection.compute_correlation(feature, target_column),
                   axis=0)

        # Select the features with the absolute correlation above the threshold
        filtered_features = [feature for feature, correlation in spearman_correlations.items()
                             if correlation >= threshold]
        return filtered_features
