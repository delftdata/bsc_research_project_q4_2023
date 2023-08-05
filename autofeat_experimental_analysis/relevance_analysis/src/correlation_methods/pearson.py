"""
Module for feature selection using Pearson method.
"""
from scipy.stats import pearsonr


class PearsonFeatureSelection:
    """
    Class to perform feature selection using the Pearson method.

    Methods
    -------
    compute_correlation(feature, target_feature): Computes the value of the correlation using Pearson
    feature_selection(train_dataframe, target_feature): Ranks all the features according to their Pearson correlation
    with the target
    """
    @staticmethod
    def compute_correlation(feature, target_feature):
        """
        Calculates the correlation between the feature and target feature using the Pearson method. It can take values
        between -1 and 1. A value of 0 means that the features are independent. A value closer to -1 means that the
        features are negatively correlated, whereas a value closer to 1 means that the features are positively
        correlated. It is a measure of linear correlation between two features.

        Parameters
        ----------
        feature (DataFrame column): Feature in the dataset
        target_feature (DataFrame column): Target feature of the dataset

        Returns
        -------
        pearson (float): Correlation between the two features measured using the Pearson method
        """
        pearson = pearsonr(feature, target_feature).statistic

        return pearson

    @staticmethod
    def feature_selection(train_dataframe, target_feature):
        """
        Performs feature selection using the Pearson correlation-based method. Ranks all the features. After calling
        this method, a specified number k of top-performing features can be selected.

        Parameters
        ----------
        train_dataframe (DataFrame): Training data containing the features
        target_feature (str): Name of the target feature column

        Returns
        -------
        selected_features (list): List of ranked features based on the Pearson correlation
        """
        target_column = train_dataframe[target_feature]
        train_dataframe = train_dataframe.drop(columns=[target_feature])

        # Calculate the Pearson correlation between each feature and the target feature
        pearson_correlations = train_dataframe \
            .apply(func=lambda feature: PearsonFeatureSelection.compute_correlation(feature, target_column),
                   axis=0)

        # Rank the features in order of the highest absolute correlation
        sorted_correlations = pearson_correlations.abs().sort_values(ascending=False)

        return sorted_correlations.index.tolist(), sorted_correlations.values.tolist()
