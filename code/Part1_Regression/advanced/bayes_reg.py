import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

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
     # Evidence Maximization 
    def evidence_maximization(self, X, y, max_iter=100, tol=1e-6):
        Phi = self._add_bias(X)
        y = y.reshape(-1, 1)

        for _ in range(max_iter):
            # posterior
            S_N_inv = self.alpha * np.eye(Phi.shape[1]) + self.beta * Phi.T @ Phi
            S_N = np.linalg.inv(S_N_inv)
            m_N = self.beta * S_N @ Phi.T @ y

            # eigenvalues
            eigenvals = np.linalg.eigvalsh(self.beta * Phi.T @ Phi)

            # gamma
            gamma = np.sum(eigenvals / (self.alpha + eigenvals))

            # update alpha, beta
            alpha_new = gamma / np.sum(m_N ** 2)

            residual = y - Phi @ m_N
            beta_new = (Phi.shape[0] - gamma) / np.sum(residual ** 2)

            # check convergence
            if abs(alpha_new - self.alpha) < tol and abs(beta_new - self.beta) < tol:
                break

            self.alpha = alpha_new
            self.beta = beta_new

        self.S_N = S_N
        self.m_N = m_N

        return self
