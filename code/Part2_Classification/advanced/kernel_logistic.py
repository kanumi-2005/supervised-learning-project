import numpy as np
from sklearn.base import ClassifierMixin, BaseEstimator


class KernelLogisticRegression(ClassifierMixin, BaseEstimator):
    def __init__(self, gamma=1.0, lr=0.1, epochs=1000):
        self.gamma = gamma
        self.lr = lr
        self.epochs = epochs

    def rbf_kernel(self, X1, X2):
        X1_sq = np.sum(X1**2, axis=1).reshape(-1, 1)
        X2_sq = np.sum(X2**2, axis=1)
        dist = X1_sq + X2_sq - 2 * X1 @ X2.T
        return np.exp(-self.gamma * dist)

    def softmax(self, Z):
        Z = Z - np.max(Z, axis=1, keepdims=True)
        expZ = np.exp(Z)
        return expZ / np.sum(expZ, axis=1, keepdims=True)

    def fit(self, X, y):
        n, d = X.shape
        self.classes = np.unique(y)
        c = len(self.classes)

        y_idx = np.array(
            [np.where(self.classes == label)[0][0] for label in y])

        self.X = X
        self.K = self.rbf_kernel(X, X)

        self.alpha = np.zeros((n, c))
        self.b = np.zeros(c)

        for _ in range(self.epochs):
            # forward
            f = self.K @ self.alpha + self.b

            p = self.softmax(f)

            # one-hot labels
            y_onehot = np.zeros_like(p)
            y_onehot[np.arange(n), y_idx] = 1

            grad_f = (p - y_onehot)  # (n, c)

            # backprop
            grad_alpha = self.K.T @ grad_f
            grad_b = np.sum(grad_f, axis=0)

            self.alpha -= self.lr * grad_alpha
            self.b -= self.lr * grad_b

    def predict(self, X):
        K_test = self.rbf_kernel(X, self.X)

        f = K_test @ self.alpha + self.b
        p = self.softmax(f)

        return self.classes[np.argmax(p, axis=1)]
