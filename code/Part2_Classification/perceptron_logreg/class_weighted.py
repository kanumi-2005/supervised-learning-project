import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from dataset import CovtypeDataset
from sklearn.preprocessing import StandardScaler
from ..base.basegdmodel import BaseGDModel


import numpy as np

class WeightedSoftmaxClassier(BaseGDModel):
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs

    # SOFTMAX
    def _softmax(self, z):
        z = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    # ONE HOT
    def _one_hot(self, y, K):
        y = np.asarray(y).astype(int)

        one_hot = np.zeros((len(y), K))
        one_hot[np.arange(len(y)), y] = 1
        return one_hot

    # CLASS WEIGHT
    def _compute_class_weights(self, y, K):
        counts = np.bincount(y, minlength=K)
        weights = len(y) / (counts + 1e-8)
        return weights

    # FIT
    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y).astype(int)

        N, D = X.shape

        self.classes_ = np.unique(y)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        y = np.array([self.class_to_idx[yy] for yy in y])

        K = len(self.classes_)

        # init params
        self.W = np.zeros((D, K))
        self.b = np.zeros(K)

        y_onehot = self._one_hot(y, K)
        class_weights = self._compute_class_weights(y, K)

        # reshape for broadcasting
        cw = class_weights.reshape(1, -1)

        for epoch in range(self.epochs):

            # forward
            logits = X @ self.W + self.b
            y_pred = self._softmax(logits)

            # loss
            loss = -np.sum(
                y_onehot * np.log(y_pred + 1e-8) * cw
            ) / N

            # gradient
            error = (y_pred - y_onehot) * cw

            dW = X.T @ error / N
            db = np.sum(error, axis=0) / N

            # update
            self.W -= self.lr * dW
            self.b -= self.lr * db

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

        return self

    def predict(self, X):
        logits = X @ self.W + self.b
        probs = self._softmax(logits)
        y_pred = np.argmax(probs, axis=1)

        # map về label gốc
        inv_map = {v: k for k, v in self.class_to_idx.items()}
        return np.array([inv_map[i] for i in y_pred])