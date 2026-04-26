import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from dataset import CaliforniaHousingDataset as Dataset
from .polynomial import PolynomialBasis
from .rbf import RBF
from .fourier import FourierBasis


class ValidationExperiment:

    def __init__(self):
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None

    # =========================
    # UTILS
    # =========================
    def compute_mse(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    # =========================
    # DATA
    # =========================
    def load_data(self):
        d = Dataset()
        d.split()

        self.X_train, self.y_train = d.X_train, d.y_train
        self.X_val, self.y_val = d.X_val, d.y_val

        self.y_train = self.y_train.ravel()
        self.y_val = self.y_val.ravel()

        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_val = scaler.transform(self.X_val)

    # =========================
    # GENERIC RUN
    # =========================
    def run(self, basis_class, param_name, param_values):
        mses = []

        for val in param_values:
            model = Pipeline([
                ("basis", basis_class(**{param_name: val})),
                ("model", LinearRegression())
            ])

            model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_val)

            mse = self.compute_mse(self.y_val, y_pred)
            mses.append(mse)

            print(f"{param_name}={val} → MSE={mse:.4f}")

        return list(param_values), mses

    # =========================
    # WRAPPERS
    # =========================
    def run_polynomial(self):
        return self.run(PolynomialBasis, "degree", range(1, 11))

    def run_rbf(self):
        return self.run(RBF, "n_centers", range(5, 105, 10))

    def run_fourier(self):
        return self.run(FourierBasis, "n_terms", range(1, 11))

    # =========================
    # PLOT
    # =========================
    def plot(self, x, y, title, xlabel):
        plt.figure()

        plt.plot(x, y, marker='o', label="Validation MSE")

        # highlight best
        best_idx = np.argmin(y)
        plt.scatter(x[best_idx], y[best_idx], s=100, label=f"Best: {x[best_idx]}")

        plt.xlabel(xlabel)
        plt.ylabel("MSE")
        plt.title(title)

        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # =========================
    # RUN ALL
    # =========================
    def run_all(self):
        # Polynomial
        x, y = self.run_polynomial()
        self.plot(x, y, "Polynomial Validation Curve", "Degree")

        # RBF
        x, y = self.run_rbf()
        self.plot(x, y, "RBF Validation Curve", "Number of Centers")

        # Fourier
        x, y = self.run_fourier()
        self.plot(x, y, "Fourier Validation Curve", "Degree")