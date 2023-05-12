from scipy.stats import pearsonr


def pearson_calculation(x, y):
    return pearsonr(x, y).statistic
