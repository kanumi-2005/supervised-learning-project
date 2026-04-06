import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from code.Part2_Classification.dataset import CovtypeDataset
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


def observe_convergence():
    import matplotlib.pyplot as plt

    # ===== 1. LINEAR DATA =====
    result = make_blobs(n_samples=200, centers=2, random_state=42)
    X_linear, y_linear = result[:2]
    y_linear = np.where(y_linear == 0, -1, 1)

    model1 = Perceptron(lr=0.01, epochs=50)
    model1.fit(X_linear, y_linear)

    # ===== 2. NON-LINEAR DATA =====
    X_non, y_non = make_circles(n_samples=200, noise=0.1, factor=0.5)
    y_non = np.where(y_non == 0, -1, 1)

    model2 = Perceptron(lr=0.01, epochs=50)
    model2.fit(X_non, y_non)

    # ===== PLOT =====
    plt.figure(figsize=(10, 8))

    plt.subplot(2, 2, 1)
    plt.scatter(X_linear[:, 0], X_linear[:, 1], c=y_linear,s=5)
    # draw decision boundary
    x_vals = np.linspace(X_linear[:, 0].min(), X_linear[:, 0].max(), 100)
    y_vals = -(model1.w[0] * x_vals + model1.b) / model1.w[1]
    plt.plot(x_vals, y_vals, linewidth=2)
    plt.title("Linear Separable Data (Blobs)")

    plt.subplot(2, 2, 2)
    plt.scatter(X_non[:, 0], X_non[:, 1], c=y_non,s=5)
    x_vals = np.linspace(X_non[:, 0].min(), X_non[:, 0].max(), 100)
    y_vals = -(model2.w[0] * x_vals + model2.b) / model1.w[1]
    plt.plot(x_vals, y_vals, linewidth=2)
    plt.title("Non-linear Data")

    plt.subplot(2, 2, 3)
    plt.plot(model1.errors_, marker='o', markersize=3, linewidth=1)
    plt.title("Linear Data (Converges)")
    plt.xlabel("Epoch")
    plt.ylabel("Error rate")
    plt.grid(alpha=0.3)

    plt.subplot(2, 2, 4)
    plt.plot(model2.errors_, marker='o', markersize=3, linewidth=1)
    plt.title("Non-linear Data (Not Converged)")
    plt.xlabel("Epoch")
    plt.ylabel("Error rate")
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    # ===== PRINT =====
    print("Linear:", "Converged" if model1.errors_[-1] == 0 else "Not converged")
    print("Non-linear:", "Converged" if model2.errors_[-1] == 0 else "Not converged")

def main():
    observe_convergence()

if __name__ == "__main__":
    main()