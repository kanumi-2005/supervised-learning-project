import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from code.Part1_Regression.dataset import CaliforniaHousingDataset as Dataset

class BayesianLinearRegression(BaseEstimator, RegressorMixin):
    def __init__(self, alpha = 1.0, beta = 10.0):
        self.alpha = alpha #prior decision 
        self.beta = beta #noise decision 
    def _add_bias(self,X):
        return np.hstack([np.ones((X.shape[0],1)),X])
    def fit(self,X,y):
        phi = self._add_bias(X)
        # S_N^{-1} = αI + βΦ^TΦ
        S_N_inv = self.alpha*np.eye(phi.shape[1]) + self.beta * phi.T @ phi
        self.S_N = np.linalg.inv(S_N_inv)
        y = y.reshape(-1, 1)
         # m_N = β S_N Φ^T y
        self.m_N = self.beta * self.S_N @ phi.T @ y
        return self 
    def predict(self,X):
        phi = self._add_bias(X)
        return phi @ self.m_N
    def predict_dist(self,X):
        """
        return predictive mean and std
        """
        phi = self._add_bias(X)

        #mean
        y_mean = phi @ self.m_N
        # variance: σ² = 1/β + Φ S_N Φ^T
        y_var = 1/ self.beta + np.sum(phi @ self.S_N * phi, axis=1)
        y_std = np.sqrt(y_var)
        return y_mean.ravel(), y_std.ravel()
    def get_posterior(self):
        return self.m_N, self.S_N

# ===== MAIN =====
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

    # ===== MODEL =====
    model = BayesianLinearRegression(alpha=1.0, beta=10.0)
    model.fit(X_train, y_train)

    # ===== POSTERIOR =====
    m_N, S_N = model.get_posterior()
    print("Posterior mean shape:", m_N.shape)
    print("Posterior covariance shape:", S_N.shape)

    # ===== PREDICT =====
    y_mean, y_std = model.predict_dist(X_test)

    # ===== EVALUATE =====
    mse = np.mean((y_test - y_mean) ** 2)
    print(f"MSE = {mse:.6f}")

    # ===== PLOT (1D visualization) =====
    X_plot = X_test[:, 0]
    idx = np.argsort(X_plot)

    plt.figure(figsize=(8, 5))

    # data test
    plt.scatter(X_plot, y_test, s=10, label="Test data")

    # mean prediction
    plt.plot(X_plot[idx], y_mean[idx], label="Mean")

    # vùng bất định ±2σ
    plt.fill_between(
        X_plot[idx],
        y_mean[idx] - 2*y_std[idx],
        y_mean[idx] + 2*y_std[idx],
        alpha=0.3,
        label="±2σ"
    )
    plt.xlabel("Median Income (MedInc) - Feature 0")
    plt.ylabel("House Price (Target)")
    plt.legend()
    plt.title("Bayesian Linear Regression (Uncertainty)")
    plt.show()

