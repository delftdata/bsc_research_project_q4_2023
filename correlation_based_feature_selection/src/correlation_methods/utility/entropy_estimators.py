"""
Module for computing the discrete estimators.
"""
from math import log


def calculate_entropy(feature_samples, base=2):
    """
    Discrete entropy estimator given a list of samples which can be any hashable object
    """
    return calculate_entropy_from_probabilities(compute_histogram(feature_samples), base=base)


def midd(feature, target_feature):
    """
    Discrete mutual information estimator given a list of samples which can be any hashable object
    """
    return -calculate_entropy(list(zip(feature, target_feature))) + \
        calculate_entropy(feature) + calculate_entropy(target_feature)


def compute_histogram(feature_samples):
    occurrences = {}
    for sample in feature_samples:
        occurrences[sample] = occurrences.get(sample, 0) + 1

    return map(lambda z: float(z) / len(feature_samples), occurrences.values())


def calculate_entropy_from_probabilities(probs, base=2):
    # Turn a normalized list of probabilities of discrete outcomes into entropy (base 2)
    return -sum(map(elog, probs)) / log(base)


def elog(value):
    if value <= 0. or value >= 1.:
        return 0
    return value * log(value)
