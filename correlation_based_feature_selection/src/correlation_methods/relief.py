"""
Module for feature selection using Relief family of feature selection methods.
"""
import sklearn_relief as relief
import pandas as pd


class ReliefFeatureSelection:
    """
    Class to perform feature selection using the Relief family of feature selection methods.

    Methods
    -------
    feature_selection(train_dataframe, target_feature): Ranks the features according to their relevance for predicting
    the target
    """
    @staticmethod
    def feature_selection(train_dataframe, target_feature, number_of_features_k,
                          problem_type='binary_classification'):
        """
        Performs feature selection using the Relief family of feature selection methods. The choice of method
        depends on the problem type: binary classification - original Relief method, multiclass classification - ReliefF
        method and regression - RReliefF method. Selects a specified number of top-performing features.

        Parameters
        ----------
        train_dataframe (DataFrame): Training data containing the features
        target_feature (str): Name of the target feature

        Returns
        -------
        selected_features (list): List of selected features using Relief/ReliefF/RReliefF
        """
        target_column = train_dataframe[target_feature]
        train_dataframe = train_dataframe.drop(columns=[target_feature])

        relief_method = relief.Relief(n_features=number_of_features_k)
        if problem_type == 'multiclass_classification':
            relief_method = relief.ReliefF(n_features=number_of_features_k)
        if problem_type == 'regression':
            relief_method = relief.RReliefF(n_features=number_of_features_k)
        transformed_train_dataframe = relief_method.fit_transform(train_dataframe.values, target_column.values)

        # Calculate the Relief weight for each feature in the dataset
        features = train_dataframe.columns.tolist()
        feature_weights = relief_method.w_

        relief_correlations = pd.DataFrame({'feature': features, 'relief_weight': feature_weights})

        # Select the top features with the highest correlation
        sorted_correlations = relief_correlations.sort_values(by='relief_weight', ascending=False)
        print(sorted_correlations)
        return sorted_correlations['feature'].tolist(), sorted_correlations['relief_weight'].tolist()
