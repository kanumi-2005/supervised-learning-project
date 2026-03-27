import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from code.Part1_Regression.nonlinear_basis.rbf import RBF
from code.Part1_Regression.nonlinear_basis.fourier import FourierBasis
from code.Part1_Regression.nonlinear_basis.polynomial import PolynomialBasis

from code.Part1_Regression.dataset import CaliforniaHousingDataset as Dataset
from code.Part1_Regression.linear_regression.wls import WLS
from code.Part1_Regression.pipeline import get_pipeline


# =========================
# LOAD DATA
# =========================
d = Dataset()
d.split()

X_train, y_train = d.X_train, d.y_train
X_test, y_test = d.X_test, d.y_test

# scale X
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =========================
# UTIL FUNCTION
# =========================
results = {}

def run_experiment(Xtr, Xte, name):
    # add bias
    Xtr = np.hstack([np.ones((Xtr.shape[0], 1)), Xtr])
    Xte = np.hstack([np.ones((Xte.shape[0], 1)), Xte])

    model = get_pipeline(LinearRegression())
    model.fit(Xtr, y_train)
    pred = model.predict(Xte)

    mse = np.mean((y_test - pred) ** 2)
    results[name] = mse


# =========================
# INIT BASE FUNCTIONS
# =========================
rbf = RBF(n_centers=20, gamma=0.1)
rbf.fit(X_train)

fourier = FourierBasis(n_terms=5)
fourier.fit(X_train)

poly = PolynomialBasis(degree=3)
poly.fit(X_train)


# =========================
# ABALATION STUDY
# =========================
# 1. Individual
run_experiment(rbf.transform(X_train), rbf.transform(X_test), "RBF")
run_experiment(fourier.transform(X_train), fourier.transform(X_test), "Fourier")
run_experiment(poly.transform(X_train), poly.transform(X_test), "Polynomial")

# 2. Remove 1 type (pairwise)
run_experiment(
    np.hstack([rbf.transform(X_train), fourier.transform(X_train)]),
    np.hstack([rbf.transform(X_test), fourier.transform(X_test)]),
    "RBF + Fourier"
)

run_experiment(
    np.hstack([rbf.transform(X_train), poly.transform(X_train)]),
    np.hstack([rbf.transform(X_test), poly.transform(X_test)]),
    "RBF + Polynomial"
)

run_experiment(
    np.hstack([fourier.transform(X_train), poly.transform(X_train)]),
    np.hstack([fourier.transform(X_test), poly.transform(X_test)]),
    "Fourier + Polynomial"
)

# 3. FULL MODEL
run_experiment(
    np.hstack([
        rbf.transform(X_train),
        fourier.transform(X_train),
        poly.transform(X_train)
    ]),
    np.hstack([
        rbf.transform(X_test),
        fourier.transform(X_test),
        poly.transform(X_test)
    ]),
    "FULL"
)


# =========================
# RESULTS
# =========================
print("\n===== ABLATION STUDY RESULTS =====")
for k, v in results.items():
    print(f"{k:25s} : {v:.6f}")


# =========================
# PLOT
# =========================
names = list(results.keys())
values = list(results.values())

plt.figure(figsize=(10, 5))
plt.bar(names, values)
plt.xticks(rotation=45)
plt.ylabel("MSE")
plt.title("Ablation Study - Basis Function Importance")
plt.grid(axis='y')
plt.tight_layout()
plt.show()