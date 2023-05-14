import numpy as np
from pandas import crosstab
from scipy.stats import contingency


def cramer_calculation(x, y):
    obs4x2 = np.array([[100, 150], [203, 322], [420, 700], [320, 210]])
    return contingency.association(x, method="cramer")
