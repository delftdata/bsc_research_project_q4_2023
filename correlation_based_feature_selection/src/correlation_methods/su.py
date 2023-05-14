from .utility.entropy_estimators import entropyd
from .utility.mutual_information import information_gain

class SymmetricUncertaintyFeatureSelection:
    @staticmethod
    def compute_correlation(feature, target_feature):
        # calculate information gain of feature and target
        # t1 = ig(feature, target_feature)
        t1 = information_gain(feature, target_feature)
        # calculate entropy of feature
        # t2 = H(feature)
        t2 = entropyd(feature)
        # calculate entropy of target
        # t3 = H(target_feature)
        t3 = entropyd(target_feature)
        # calculate the symmetric uncertainty between feature and target feature
        # su(feature, target_feature) = 2 * t1 / (t2 + t3)
        symmetric_uncertainty = 2.0 * t1 / (t2 + t3)

        return symmetric_uncertainty

    @staticmethod
    def feature_selection(dataframe, target_feature):
        # TODO: add the functionality
        return dataframe, target_feature
