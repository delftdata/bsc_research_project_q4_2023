"""
Module for feature selection using Relief family of feature selection methods.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import euclidean_distances


class ReliefFeatureSelection:
    """
    Class to perform feature selection using the Relief family of feature selection methods.

    Methods
    -------
    feature_selection(train_dataframe, target_feature): Ranks the features according to their relevance for predicting
    the target
    """
    def knn_from_class(distances, y, index, k, cl, anyOtherClass=False, anyClass=False):
        """
        Returns the indices of k nearest neighbors of X[index] from the selected class. Source of the code is:
        https://github.com/ctlab/ITMO_FS/tree/a2e61e2fabb9dfb34d90a1130fc7f5f162a2c921.

        Parameters
        ----------
        distances (array-like with shape (n_samples, n_samples)): The distance matrix of the input samples
        y (array-like with shape (n_samples,)): The classes for the samples
        index (int): The index of an element
        k (int): The amount of nearest neighbors to return
        cl (int): The class label for the nearest neighbors
        anyClass (bool): If True, returns neighbors not belonging to the same class as X[index]

        Returns
        -------
        (array-like with shape (k,)): The indices of the nearest neighbors
        """
        y_c = np.copy(y)
        if anyOtherClass:
            cl = y_c[index] + 1
            y_c[y_c != y_c[index]] = cl
        if anyClass:
            y_c.fill(cl)
        class_indices = np.nonzero(y_c == cl)[0]
        distances_class = distances[index][class_indices]
        nearest = np.argsort(distances_class)
        if y_c[index] == cl:
            nearest = nearest[1:]

        return class_indices[nearest[:k]]

    @staticmethod
    def relief_measure(x, y, m=None, random_state=42):
        """
        Calculates Relief measure for each feature. This measure is supposed to work only with binary classification
        datasets; for multi-class problems use the ReliefF measure. Bigger values mean more important features. Source
        of the code is: https://github.com/ctlab/ITMO_FS/tree/a2e61e2fabb9dfb34d90a1130fc7f5f162a2c921.

        Parameters
        ----------
        x (array-like with shape (n_samples, n_features)): The input samples
        y (array-like with shape (n_samples,)): The classes for the samples
        m (int, optional): Amount of iterations to do. If not specified, n_samples iterations would be performed
        random_state (int, optional): Random state for numpy random

        Returns
        -------
        (array-like with shape (n_features,)): Feature scores
        """
        weights = np.zeros(x.shape[1])
        classes, counts = np.unique(y, return_counts=True)
        if len(classes) == 1:
            raise ValueError("Cannot calculate relief measure with 1 class")
        if 1 in counts:
            raise ValueError(
                "Cannot calculate relief measure because one of the classes has "
                "only 1 sample")

        n_samples = x.shape[0]
        n_features = x.shape[1]
        if m is None:
            m = n_samples

        x_normalized = MinMaxScaler().fit_transform(x)
        dm = euclidean_distances(x_normalized, x_normalized)
        indices = np.random.default_rng(random_state).integers(
            low=0, high=n_samples, size=m)
        objects = x_normalized[indices]
        hits_diffs = np.square(
            np.vectorize(
                lambda index: (
                        x_normalized[index]
                        - x_normalized[ReliefFeatureSelection.knn_from_class(dm, y, index, 1, y[index])]),
                signature='()->(n,m)')(indices))
        misses_diffs = np.square(
            np.vectorize(
                lambda index: (
                        x_normalized[index]
                        - x_normalized[ReliefFeatureSelection.knn_from_class(dm, y, index, 1, y[index], anyOtherClass=True)]
                ),
                signature='()->(n,m)')(indices))

        H = np.sum(hits_diffs, axis=(0, 1))
        M = np.sum(misses_diffs, axis=(0, 1))

        weights = M - H

        return weights / m

    @staticmethod
    def feature_selection(train_dataframe, target_feature, number_of_features_k):
        """
        Performs feature selection using the Relief family of feature selection methods. Ranks all the features.
        Selects the top-performing k features.

        The choice of method depends on the problem type: binary classification - original Relief method, multiclass
        classification - ReliefF method and regression - RReliefF method. This method currently supports only binary
        classification datasets.

        Parameters
        ----------
        train_dataframe (DataFrame): Training data containing the features
        target_feature (str): Name of the target feature

        Returns
        -------
        selected_features (list): List of selected features using Relief
        """
        target_column = train_dataframe[target_feature]
        train_dataframe = train_dataframe.drop(columns=[target_feature])

        # Calculate the Relief value between each feature and the target feature
        features = train_dataframe.columns.tolist()
        feature_scores = ReliefFeatureSelection.relief_measure(train_dataframe.values, target_column.values)
        relief_correlations = pd.DataFrame({'feature': features, 'relief_weight': feature_scores})

        # Rank the features in order of the value returned by Relief
        sorted_correlations = relief_correlations.sort_values(by='relief_weight', ascending=False)

        return sorted_correlations['feature'].tolist(), sorted_correlations['relief_weight'].tolist()
