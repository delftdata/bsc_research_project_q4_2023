from scipy.stats import spearmanr


def spearman_calculation(x, y):
    return spearmanr(x, y).statistic
