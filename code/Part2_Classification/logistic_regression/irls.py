import time
import numpy as np
from sklearn.base import ClassifierMixin, BaseEstimator


def sigmoid(x):
    x = np.clip(x, -50, 50)
    return 1 / (1 + np.exp(-x))


class IRLSClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, max_iter=20, verbose=False, random_state=42):
        self.max_iter = max_iter
        self.verbose = verbose
        self.random_state = random_state
        self.loss_history = []
        self.time_history = []

    def _encode_labels(self, y):
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("IRLSClassifier only supports binary")
        return (y == self.classes_[1]).astype(float)

    def _loss(self, p, y):
        eps = 1e-12
        return -np.mean(y * np.log(p + eps) +
                        (1 - y) * np.log(1 - p + eps))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        y_bin = self._encode_labels(y)

        X_design = np.c_[np.ones(n_samples), X]
        w = np.zeros(n_features + 1)

        self.loss_history = []
        self.time_history = []
        start_time = time.time()

        for i in range(self.max_iter):
            z = X_design @ w
            p = sigmoid(z)

            loss = self._loss(p, y_bin)
            elapsed = time.time() - start_time

            if self.verbose:
                self.loss_history.append(loss)
                self.time_history.append(elapsed)

            W = p * (1 - p)
            H = X_design.T @ (W[:, None] * X_design)
            g = X_design.T @ (p - y_bin)

            try:
                delta = np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                delta = np.linalg.pinv(H) @ g

            w = w - delta

        self.intercept_ = w[0]
        self.coef_ = w[1:]
        return self

    def predict_proba(self, X):
        p = sigmoid(self.intercept_ + X @ self.coef_)
        return np.c_[1 - p, p]

    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]
