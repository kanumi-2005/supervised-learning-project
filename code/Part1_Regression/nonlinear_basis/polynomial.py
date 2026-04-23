import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from ..base.basemodel import BaseModel


class PolynomialBasis(BaseModel, TransformerMixin):
    def __init__(self, degree=5):
        self.degree = degree

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X)
        features = [X]

        for d in range(2, self.degree + 1):
            features.append(X ** d)

        return np.concatenate(features, axis=1)