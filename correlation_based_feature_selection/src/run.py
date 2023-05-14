from .correlation_methods.pearson import PearsonFeatureSelection
from .correlation_methods.spearman import SpearmanFeatureSelection
from .correlation_methods.cramer import CramersVFeatureSelection
from .correlation_methods.su import SymmetricUncertaintyFeatureSelection

if __name__ == '__main__':
    a = [1, 2, 3, 3]
    b = [1000, 7, 6, 1000]
    # a = ['a', 'b', 'c', 'a', 'a', 'b']
    # b = ['ddfdd', 'dsdfse', 'sdfsdf', 'sdfsf', 'coobbbi', 'coobbbi']
    print(PearsonFeatureSelection.compute_correlation(a, b))
    print(SpearmanFeatureSelection.compute_correlation(a, b))
    print(CramersVFeatureSelection.compute_correlation(a, b))
    print(SymmetricUncertaintyFeatureSelection.compute_correlation(a, b))
