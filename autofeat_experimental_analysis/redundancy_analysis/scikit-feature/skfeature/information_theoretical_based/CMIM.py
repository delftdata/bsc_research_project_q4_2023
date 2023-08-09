import numpy as np

from skfeature.utility.entropy_estimators import *

import random
import time

from scipy.stats import rankdata


def reverse_argsort(X, size=None):
    """
    This function takes the indexes of features (0 being most important -1 being least)
    and converts them to a rank based system aligned with `sklearn.SelectKBest`

    Input
    -----
    X: {numpy array} shape(n_features, ), the indices of the feature with the first
        element being most important and the last element being the least important

    Output
    ------
    F: {numpy array} ranking of the feature indices that are sklearn friendly

    """

    def dedup(seq):
        """
        Based on uniqifiers benchmarks
        https://stackoverflow.com/questions/480214/how-do-you-remove-duplicates-from-a-list-in-whilst-preserving-order
        """
        seen = set()
        seen_add = seen.add
        return [x for x in seq if not (x in seen or seen_add(x))]

    X = dedup(list(X))
    if size is None:
        X = np.array(X)
        return np.array(rankdata(-X) - 1, dtype=int)

    else:
        # else we have to pad it with the values...
        X_all = list(range(size))
        X_raw = X[:]
        X_unseen = [x for x in X_all if x not in X_raw]
        X_unseen = list(set(X_unseen))
        X_unseen_shuffle = random.sample(X_unseen[:], len(X_unseen))
        X_obj = X_raw + X_unseen_shuffle
        return reverse_argsort(X_obj[:])


def cmim(X, y, mode="index", **kwargs):
    """
    This function implements the CMIM feature selection.
    The scoring criteria is calculated based on the formula j_cmim=I(f;y)-max_j(I(fj;f)-I(fj;f|y))

    Input
    -----
    X: {numpy array}, shape (n_samples, n_features)
        Input data, guaranteed to be a discrete numpy array
    y: {numpy array}, shape (n_samples,)
        guaranteed to be a numpy array
    kwargs: {dictionary}
        n_selected_features: {int}
            number of features to select

    Output
    ------
    F: {numpy array}, shape (n_features,)
        index of selected features, F[0] is the most important feature
    J_CMIM: {numpy array}, shape: (n_features,)
        corresponding objective function value of selected features
    MIfy: {numpy array}, shape: (n_features,)
        corresponding mutual information between selected features and response

    Reference
    ---------
    Brown, Gavin et al. "Conditional Likelihood Maximisation: A Unifying Framework for Information Theoretic Feature Selection." JMLR 2012.
    """

    n_samples, n_features = X.shape
    times = []
    # index of selected features, initialized to be empty
    F = []
    # Objective function value for selected features
    J_CMIM = []
    # Mutual information between feature and response
    MIfy = []
    # indicate whether the user specifies the number of features
    is_n_selected_features_specified = False

    if "n_selected_features" in list(kwargs.keys()):
        n_selected_features = kwargs["n_selected_features"]
        is_n_selected_features_specified = True

    # t1 stores I(f;y) for each feature f
    t1 = np.zeros(n_features)

    # max stores max(I(fj;f)-I(fj;f|y)) for each feature f
    # we assign an extreme small value to max[i] ito make it is smaller than possible value of max(I(fj;f)-I(fj;f|y))
    max = -10000000 * np.ones(n_features)

    time_before = time.time()

    for i in range(n_features):
        f = X[:, i]
        t1[i] = midd(f, y)

    # make sure that j_cmi is positive at the very beginning
    j_cmim = 1

    while True:
        if len(F) == 0:
            # select the feature whose mutual information is the largest
            idx = np.argmax(t1)
            F.append(idx)
            J_CMIM.append(t1[idx])
            MIfy.append(t1[idx])
            f_select = X[:, idx]
            times.append(time.time() - time_before)

        if is_n_selected_features_specified:
            print(len(F))
            if len(F) == n_selected_features:
                break
        else:
            if j_cmim <= 0:
                break

        # we assign an extreme small value to j_cmim to ensure it is smaller than all possible values of j_cmim
        j_cmim = -1000000000000
        for i in range(n_features):
            if i not in F:
                f = X[:, i]
                t2 = midd(f_select, f)
                t3 = cmidd(f_select, f, y)
                if t2 - t3 > max[i]:
                    max[i] = t2 - t3
                # calculate j_cmim for feature i (not in F)
                t = t1[i] - max[i]
                # record the largest j_cmim and the corresponding feature index
                if t > j_cmim:
                    j_cmim = t
                    idx = i
        F.append(idx)
        J_CMIM.append(j_cmim)
        MIfy.append(t1[idx])
        f_select = X[:, idx]
        times.append(time.time() - time_before)
    if mode == "index":
        return np.array(F), np.array(J_CMIM), np.array(MIfy), np.array(times)
    else:
        return reverse_argsort(F), np.array(J_CMIM), np.array(MIfy), np.array(times)