import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

from code.Part1_Regression.dataset import CaliforniaHousingDataset as Dataset
from code.Part1_Regression.linear_regression.wls import WLS
from code.Part1_Regression.pipeline import get_pipeline


# FOURIER BASIS FUNCTION
class FourierBasis:
    def __init__(self, n_terms=5):
        self.n_terms = n_terms

    def fit(self, X):
        return self

    def transform(self, X):
        features = []

        for k in range(1, self.n_terms + 1):
            features.append(np.sin(k * X))
            features.append(np.cos(k * X))

        return np.concatenate(features, axis=1)


# MAIN
if __name__ == "__main__":

    # ===== LOAD DATA =====
    d = Dataset()
    d.split()

    X_train, y_train = d.X_train, d.y_train
    X_test, y_test = d.X_test, d.y_test

    # ===== SCALE X =====
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # ===== FOURIER TRANSFORM =====
    fourier = FourierBasis(n_terms=5)
    fourier.fit(X_train)

    X_train_f = fourier.transform(X_train)
    X_test_f = fourier.transform(X_test)
    X_train_f = np.hstack([np.ones((X_train_f.shape[0], 1)), X_train_f])
    X_test_f = np.hstack([np.ones((X_test_f.shape[0], 1)), X_test_f])

    # ===== MODEL =====
    model = LinearRegression()
    model.fit(X_train_f, y_train)
    y_pred = model.predict(X_test_f)

    # ===== EVALUATE =====
    mse = np.mean((y_test - y_pred) ** 2)
    print(f"MSE         : {mse:.6f}")
    print("=" * 45)