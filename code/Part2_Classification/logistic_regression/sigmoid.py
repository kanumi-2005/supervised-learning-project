import numpy as np
from numpy._core.numeric import indices
from sklearn.base import ClassifierMixin, BaseEstimator


def sigmoid(x):
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


    def _lr(self, iteration):
        if self.lr_sched is None:
            return self.lr
        elif self.lr_sched == "step_decay":
            return self.lr * (self.decay_factor ** \
                   (iteration // self.step_size))
        else:
            raise ValueError(f"Unknown lr_sched: {self.lr_sched}")

    def fit(self, X, y):
        rng = np.random.default_rng(self.random_state)

        n_samples, n_features = X.shape
        X_design = np.c_[np.ones(X.shape[0]), X]
        weights = np.zeros(n_features + 1)
        for _ in range(self.max_iter):
            indices = rng.permutation(n_samples)
            X_shuffled = X_design[indices]
            y_shuffled = y[indices]

            for start in range(0, n_samples, self.batch_size):
                end = start + self.batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                y_pred_proba = sigmoid(X_batch @ weights)
                grad = (X_batch.T @ (y_pred_proba - y_batch)) / \
                       X_batch.shape[0]

                weights -= self.lr * grad

        self.intercept_ = weights[0]
        self.coef_  = weights[1:]
        return self

    def predict_proba(self, X):
        return sigmoid(self.intercept_ + X @ self.coef_)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)
