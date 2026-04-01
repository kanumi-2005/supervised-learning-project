import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
from code.Part1_Regression.dataset import CaliforniaHousingDataset as Dataset
from code.Part1_Regression.advanced.bayes_reg import BayesianLinearRegression
import time


def compare_em_vs_cv(X_train, y_train, X_test, y_test):

    # EM 
    start = time.time()

    model_em = BayesianLinearRegression()
    model_em.evidence_maximization(X_train, y_train)

    y_pred_em = model_em.predict(X_test)

    em_time = time.time() - start
    em_mse = mean_absolute_error(y_test, y_pred_em)

    #  CV 
    param_grid = {
        "alpha": [0.1, 1, 10],
        "beta": [1, 10, 100]
    }

    start = time.time()

    grid = GridSearchCV(
        BayesianLinearRegression(),
        param_grid,
        cv=3
    )
    grid.fit(X_train, y_train)

    y_pred_cv = grid.predict(X_test)

    cv_time = time.time() - start
    cv_mse = mean_absolute_error(y_test, y_pred_cv)

    # ================= RESULT =================
    print("\n===== EVIDENCE MAXIMIZATION =====")
    print("alpha:", model_em.alpha)
    print("beta:", model_em.beta)
    print("MSE:", em_mse)
    print("Time:", em_time)

    print("\n===== CROSS VALIDATION =====")
    print("Best params:", grid.best_params_)
    print("MSE:", cv_mse)
    print("Time:", cv_time)

    print("\n===== COMPARISON =====")
    print(f"EM faster than CV? {em_time < cv_time}")
    print(f"EM better MSE than CV? {em_mse < cv_mse}")


# ===== MAIN =====
if __name__ == "__main__":

    from code.Part1_Regression.dataset import CaliforniaHousingDataset as Dataset
    from sklearn.preprocessing import StandardScaler

    # Load data
    d = Dataset()
    d.split()

    X_train, y_train = d.X_train, d.y_train
    X_test, y_test = d.X_test, d.y_test

    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Compare
    compare_em_vs_cv(X_train, y_train, X_test, y_test)