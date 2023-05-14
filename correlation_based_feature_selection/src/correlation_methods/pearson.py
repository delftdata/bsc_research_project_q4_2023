from scipy.stats import pearsonr


class PearsonFeatureSelection:
    @staticmethod
    def compute_correlation(feature, target_feature):
        return pearsonr(feature, target_feature).statistic

    @staticmethod
    def feature_selection(dataframe, target_feature):
        # TODO: add the functionality
        return dataframe, target_feature
