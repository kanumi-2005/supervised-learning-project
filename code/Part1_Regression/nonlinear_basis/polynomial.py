import numpy as np

# POLYNOMIAL BASIS
class PolynomialBasis:
    def __init__(self, degree=5):
        self.degree = degree

    def fit(self, X):
        return self

    def transform(self, X):
        features = [X]

        for d in range(2, self.degree + 1):
            features.append(X ** d)

        return np.concatenate(features, axis=1)

