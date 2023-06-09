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
        SELECT K BEST ALGORITHM: Calculates the correlation between the feature and target feature using Cramer's V
        method. It is computed by taking the square root of the chi-squared statistic divided by the number
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
    def feature_selection(train_dataframe, target_feature, number_features):
        """
        Performs feature selection using the Cramer's V correlation-based method. Selects
        a specified number of top-performing features.

        Parameters
        ----------
        train_dataframe (DataFrame): Training data containing the features
        target_feature (str): Name of the target feature column
        number_features (int): Number of best-performing features to select

        Returns
        -------
        selected_features (list): List of selected feature names based on Cramer's V correlation
        """
        target_column = train_dataframe[target_feature]
        train_dataframe = train_dataframe.drop(columns=[target_feature])

        # Calculate the Cramer's V correlation between each feature and the target feature
        cramersv_correlations = train_dataframe\
            .apply(func=lambda feature: CramersVFeatureSelection.compute_correlation(feature, target_column),
                   axis=0)

        # Select the top features with the highest correlation
        sorted_correlations = cramersv_correlations.sort_values(ascending=False)

        return sorted_correlations[:number_features].index.tolist()

    @staticmethod
    def feature_selection_second_approach(train_dataframe, target_feature, threshold):
        """
        SELECT ABOVE C ALGORITHM: Performs feature selection using Cramer's V correlation-based method.
        Selects a number of features that have correlation with the target above a certain threshold.

        Parameters
        ----------
        train_dataframe (DataFrame): Training data containing the features
        target_feature (str): Name of the target feature column
        threshold (float): Minimum value for the feature to be considered useful for predicting the target

        Returns
        -------
        selected_features (list): List of selected features based on the Cramer's V correlation using "Select above c"
        """
        target_column = train_dataframe[target_feature]
        train_dataframe = train_dataframe.drop(columns=[target_feature])

        # Calculate the Spearman correlation between each feature and the target feature
        cramersv_correlations = train_dataframe \
            .apply(func=lambda feature: CramersVFeatureSelection.
                   compute_correlation(feature, target_column),
                   axis=0)

        # Select the features with the absolute correlation above the threshold
        filtered_features = [feature for feature, correlation in cramersv_correlations.items()
                             if correlation >= threshold]
        return filtered_features
