import numpy as np
from pandas import crosstab
from scipy.stats import chi2_contingency


def cramer_calculation(x, y):
    contingency_table_values = crosstab(x, y).values

    chi_squared_statistic = chi2_contingency(contingency_table_values, correction=False)[0]
    total_observations = contingency_table_values.sum().sum()
    degrees_of_freedom = min(contingency_table_values.shape) - 1

    cramers_v = np.sqrt(chi_squared_statistic / (total_observations * degrees_of_freedom))

    return cramers_v
