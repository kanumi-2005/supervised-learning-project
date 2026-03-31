import numpy as np
from sklearn.base import ClassifierMixin, BaseEstimator


def sigmoid(x):
    x = np.clip(x, -50, 50)
    return 1 / (1 + np.exp(-x))


class SigmoidClassifier(ClassifierMixin, BaseEstimator):

    def __init__(
        self,
        lr=0.01,
        max_iter=150,
        batch_size=64,
        lr_sched="step_decay",
        step_size=50,
        decay_factor=0.5,
        random_state=42
    ):
        self.lr = lr
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.lr_sched = lr_sched
        self.step_size = step_size
        self.decay_factor = decay_factor
        self.random_state = random_state

    def _encode_labels(self, y):
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("SigmoidClassifier only supports binary")
        return (y == self.classes_[1]).astype(float)

    def _lr(self, iteration):
        if self.lr_sched is None:
            return self.lr
        elif self.lr_sched == "step_decay":
            return self.lr * (self.decay_factor **
                              (iteration // self.step_size))
        else:
            raise ValueError(f"Unknown lr_sched: {self.lr_sched}")

    def _gradient(self, X_batch, y_batch, w):
        p = sigmoid(X_batch @ w)
        grad = X_batch.T @ (p - y_batch)
        return grad / X_batch.shape[0]

    def fit(self, X, y):
        rng = np.random.default_rng(self.random_state)

        n_samples, n_features = X.shape
        y_bin = self._encode_labels(y)

        X_design = np.c_[np.ones(n_samples), X]
        w = np.zeros(n_features + 1)

        for i in range(self.max_iter):
            indices = rng.permutation(n_samples)
            X_shuffled = X_design[indices]
            y_shuffled = y_bin[indices]

            for start in range(0, n_samples, self.batch_size):
                end = start + self.batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                grad = self._gradient(X_batch, y_batch, w)
                w -= self._lr(i) * grad

        self.intercept_ = w[0]
        self.coef_ = w[1:]
        return self

    def predict_proba(self, X):
        p = sigmoid(self.intercept_ + X @ self.coef_)
        return np.c_[1 - p, p]

    def predict(self, X):
        probs = self.predict_proba(X)
        indices = np.argmax(probs, axis=1)
        return self.classes_[indices]
