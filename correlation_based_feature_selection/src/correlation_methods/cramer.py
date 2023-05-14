from pandas import crosstab
from scipy.stats import contingency


class CramersVFeatureSelection:
    @staticmethod
    def compute_correlation(feature, target_feature):
        contingency_table_values = crosstab(feature, target_feature).values
        return contingency.association(contingency_table_values, method="cramer")

    @staticmethod
    def feature_selection(dataframe, target_feature):
        # TODO: add the functionality
        return dataframe, target_feature
