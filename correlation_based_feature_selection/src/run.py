from .correlation_methods.su import SymmetricUncertaintyFeatureSelection

if __name__ == '__main__':
    print('hello')

    a = [1, 2, 3]
    b = [1, 2, 3]
    print(SymmetricUncertaintyFeatureSelection.compute_correlation(a, b))

