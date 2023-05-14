"""
Module for computing the mutual information gain.
"""
from .entropy_estimators import calculate_entropy, midd


def calculate_information_gain(feature, target_feature):
    information_gain = calculate_entropy(feature) - calculate_conditional_entropy(feature, target_feature)

    return information_gain


def calculate_conditional_entropy(feature, target_feature):
    conditional_entropy = calculate_entropy(feature) - midd(feature, target_feature)

    return conditional_entropy
