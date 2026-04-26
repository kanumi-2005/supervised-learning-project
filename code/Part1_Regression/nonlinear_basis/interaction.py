import numpy as np
from itertools import combinations
from sklearn.base import BaseEstimator, TransformerMixin
from base.basemodel import BaseModel


class InteractionBasis(BaseModel, TransformerMixin):
    """
    Tạo các đặc trưng tương tác dạng x_i * x_j

    Parameters
    ----------
    degree : int (default=2)
        - 2: chỉ interaction bậc 2 (x_i * x_j)
        - 3: thêm cả x_i * x_j * x_k (nếu muốn)
    include_self : bool (default=False)
        - False: chỉ lấy i < j
        - True: cho phép x_i * x_i (tức là x^2)
    """

    def __init__(self, degree=2, include_self=False):
        self.degree = degree
        self.include_self = include_self

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        n_samples, n_features = X.shape
        features = []

        # luôn giữ original features
        features.append(X)

        # tạo interaction
        for d in range(2, self.degree + 1):
            if self.include_self:
                comb = combinations(range(n_features), d)
            else:
                comb = combinations(range(n_features), d)

            for idx in comb:
                new_feature = np.prod(X[:, idx], axis=1).reshape(-1, 1)
                features.append(new_feature)

        return np.concatenate(features, axis=1)