import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from dataset import CovtypeDataset
from sklearn.datasets import make_blobs, make_circles

class Perceptron:
    def __init__(self, lr=0.01, epochs=100):
        # Learning rate
        self.lr = lr
        # Number of training epochs
        self.epochs = epochs

    def fit(self, X, y):
        # Initialize weights and bias
        self.w = np.zeros(X.shape[1])
        self.b = 0

        # Store number of errors p  er epoch
        self.errors_ = []

        # Training loop
        for _ in range(self.epochs):
            errors = 0

            for xi, yi in zip(X, y):
                # Compute linear output
                linear_output = np.dot(xi, self.w) + self.b

                # Apply step function (activation)
                y_pred = 1 if linear_output >= 0 else 0

                if yi * linear_output <= 0:
                    self.w += self.lr * yi * xi
                    self.b += self.lr * yi
                    errors += 1

            # Save error count for this epoch
            self.errors_.append(errors)

        return self

    def predict(self, X):
        # Compute linear output
        linear_output = np.dot(X, self.w) + self.b

        # Apply step function to get predictions
        return np.where(linear_output >= 0, 1, -1)

