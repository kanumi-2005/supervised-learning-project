import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans

class RBF(BaseEstimator, TransformerMixin):
    def __init__(self, n_centers=10, gamma=0.1, seed=42, include_bias=False):
        self.n_centers = n_centers
        self.gamma = gamma
        self.seed = seed
        self.include_bias = include_bias

    def fit(self, X, y=None):
        self.kmeans_ = KMeans(
            n_clusters=self.n_centers,
            random_state=self.seed
        )
        self.kmeans_.fit(X)
        self.centers_ = self.kmeans_.cluster_centers_
        return self

    def transform(self, X):
        # (n_samples, n_centers)
        diff = X[:, None, :] - self.centers_[None, :, :]
        dist_sq = np.sum(diff ** 2, axis=2)

        features = np.exp(-self.gamma * dist_sq)

        # optional bias
        if self.include_bias:
            bias = np.ones((features.shape[0], 1))
            features = np.hstack([bias, features])

        return features