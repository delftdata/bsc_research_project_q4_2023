from .correlation_methods.su import SymmetricUncertaintyFeatureSelection
from .correlation_methods.cramer import CramersVFeatureSelection

if __name__ == '__main__':
    print('hello')

    # a = [1, 2, 3]
    # b = [1, 2, 3]
    a = ['a', 'b', 'c']
    b = ['a', 'b', 'c']
    print(SymmetricUncertaintyFeatureSelection.compute_correlation(a, b))
    print(CramersVFeatureSelection.compute_correlation(a, b))
