import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

from code.Part1_Regression.dataset import CaliforniaHousingDataset as Dataset
from code.Part1_Regression.nonlinear_basis.polynomial import PolynomialBasis
from code.Part1_Regression.nonlinear_basis.rbf import RBF
from code.Part1_Regression.nonlinear_basis.fourier import FourierBasis
from code.Part1_Regression.pipeline import get_pipeline

# =========================
# UTILS
# =========================
def compute_mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# =========================
# POLYNOMIAL VALIDATION CURVE
# =========================
def run_polynomial(X_train, y_train, X_test, y_test):
    degrees = list(range(1, 11))
    mse_list = []

    for deg in degrees:
        poly = PolynomialBasis(degree=deg)

        X_train_p = poly.transform(X_train)
        X_test_p = poly.transform(X_test)

        model = get_pipeline(LinearRegression())  # KHÔNG cần bias tay
        model.fit(X_train_p, y_train)

        pred = model.predict(X_test_p)
        mse = compute_mse(y_test, pred)

        mse_list.append(mse)
        print(f"[Polynomial] Degree {deg}: MSE = {mse:.4f}")

    return degrees, mse_list


# =========================
# RBF VALIDATION CURVE
# =========================
def run_rbf(X_train, y_train, X_test, y_test):
    centers_list = [5, 10, 20, 30, 50]
    mse_list = []

    for k in centers_list:
        rbf = RBF(n_centers=k)
        rbf.fit(X_train)

        X_train_r = rbf.transform(X_train)
        X_test_r = rbf.transform(X_test)

        model = get_pipeline(LinearRegression())
        model.fit(X_train_r, y_train)

        pred = model.predict(X_test_r)
        mse = compute_mse(y_test, pred)

        mse_list.append(mse)
        print(f"[RBF] Centers {k}: MSE = {mse:.4f}")

    return centers_list, mse_list


# =========================
# FOURIER VALIDATION CURVE
# =========================
def run_fourier(X_train, y_train, X_test, y_test):
    degrees = [1, 2, 3, 5, 7, 10]
    mse_list = []

    for d in degrees:
        fourier = FourierBasis(d)

        X_train_f = fourier.transform(X_train)
        X_test_f = fourier.transform(X_test)

        model = get_pipeline(LinearRegression())
        model.fit(X_train_f, y_train)

        pred = model.predict(X_test_f)
        mse = compute_mse(y_test, pred)

        mse_list.append(mse)
        print(f"[Fourier] Degree {d}: MSE = {mse:.4f}")

    return degrees, mse_list


# =========================
# PLOT
# =========================
def plot_all(poly_x, poly_y, rbf_x, rbf_y, fourier_x, fourier_y):
    plt.figure()

    plt.plot(poly_x, poly_y, marker='o', label="Polynomial")
    plt.plot(rbf_x, rbf_y, marker='s', label="RBF")
    plt.plot(fourier_x, fourier_y, marker='^', label="Fourier")

    plt.xlabel("Model Complexity")
    plt.ylabel("MSE")
    plt.title("Validation Curve: Basis Functions Comparison")
    plt.legend()
    plt.grid(True)

    plt.show()


# =========================
# MAIN
# =========================
if __name__ == "__main__":

    # ===== LOAD DATA =====
    d = Dataset()
    d.split()

    X_train, y_train = d.X_train, d.y_train
    X_test, y_test = d.X_test, d.y_test

    # ===== SCALE =====
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # ===== RUN =====
    poly_x, poly_y = run_polynomial(X_train, y_train, X_test, y_test)
    rbf_x, rbf_y = run_rbf(X_train, y_train, X_test, y_test)
    fourier_x, fourier_y = run_fourier(X_train, y_train, X_test, y_test)

    # ===== PLOT =====
    plot_all(poly_x, poly_y, rbf_x, rbf_y, fourier_x, fourier_y)