import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class GDA(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=None, kernel=None, kernel_params=None):
        self.n_components = n_components
        self.kernel = kernel
        self.kernel_params = kernel_params
        self.y = None

    def fit(self, X, y):
        print("X fit", X.shape)
        self.classes_ = np.unique(y)
        self.class_means_ = []
        self.class_scatters_ = []

        for c in self.classes_:
            X_c = X[y == c]
            class_mean = np.mean(X_c, axis=0)
            self.class_means_.append(class_mean)
            class_scatter = np.cov(X_c.T)
            self.class_scatters_.append(class_scatter)

        self.total_scatter_ = np.cov(X.T)

        self.y = y  # Store y as an instance variable

        return self

    def transform(self, X):
        if self.n_components is None:
            self.n_components = X.shape[1]

        assert self.n_components <= X.shape[1], "Number of components cannot exceed the input feature dimension"

        eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(self.total_scatter_).dot(self.between_scatter_matrix()))

        # Sort eigenvalues and eigenvectors in descending order
        eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))]
        eig_pairs.sort(key=lambda x: x[0], reverse=True)

        # Select the top n_components eigenvectors
        eig_vecs_selected = np.array([eig_pair[1] for eig_pair in eig_pairs[:self.n_components]])
        eig_vecs_selected = eig_vecs_selected.T

        if self.kernel is not None:
            X_transformed = self._apply_kernel(X)
            eig_vecs_selected = eig_vecs_selected[:, :X_transformed.shape[1]]
        else:
            X_transformed = X

        # Perform dot product with updated eig_vecs_selected
        X_transformed = X_transformed.dot(eig_vecs_selected.T)

        return X_transformed

    def between_scatter_matrix(self):
        n_features = len(self.class_means_[0])
        between_scatter = np.zeros((n_features, n_features))

        total_mean = np.mean(self.class_means_, axis=0)

        for c in range(len(self.classes_)):
            class_mean = self.class_means_[c]
            n_samples = np.sum(self.y == self.classes_[c])
            print(class_mean.shape)
            print("total mean", total_mean.shape)
            between_scatter += n_samples * np.outer(class_mean - total_mean, class_mean - total_mean)

        return between_scatter

    def _apply_kernel(self, X):
        if self.kernel == "linear":
            return X
        elif self.kernel == "poly":
            degree = self.kernel_params.get("degree", 3)
            gamma = self.kernel_params.get("gamma", 1)
            coef0 = self.kernel_params.get("coef0", 0)
            return (gamma * X.dot(X.T) + coef0) ** degree
        elif self.kernel == "rbf":
            gamma = self.kernel_params.get("gamma", 1)
            return np.exp(-gamma * np.linalg.norm(X[:, np.newaxis] - X, axis=2) ** 2)
        else:
            raise ValueError("Invalid kernel. Available options are: linear, poly, rbf")

