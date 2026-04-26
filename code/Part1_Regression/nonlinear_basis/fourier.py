from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from regression.base.basemodel import BaseModel
class FourierBasis(BaseModel,TransformerMixin):
    def __init__(self, n_terms=5, include_bias=False):
        self.n_terms = n_terms
        self.include_bias = include_bias

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        features = []
        if self.include_bias:
            features.append(np.ones((X.shape[0], 1)))

        for k in range(1, self.n_terms + 1):
            features.append(np.sin(k * X))
            features.append(np.cos(k * X))

        return np.concatenate(features, axis=1)