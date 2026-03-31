import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    x = np.clip(x, -50, 50)
    return 1 / (1 + np.exp(-x))


class LaplaceApprox:
    def __init__(self, model):
        self.model = model

    def get_w_map(self):
        return np.r_[self.model.intercept_, self.model.coef_.ravel()]

    def compute_hessian(self, X):
        n = X.shape[0]
        X_design = np.c_[np.ones(n), X]

        w_map = self.get_w_map()
        z = X_design @ w_map
        p = sigmoid(z)

        W_diag_ = p * (1 - p)
        H = X_design.T @ (W_diag_[:, None] * X_design)

        return H, X_design

    def fit(self, X):
        self.H, _ = self.compute_hessian(X)
        self.cov_ = np.linalg.pinv(self.H)
        return self

    def decision_boundary_sigma(self, X, y, ax=None):
        x_min, x_max = X[:,0].min()-1, X[:,0].max()+1
        y_min, y_max = X[:,1].min()-1, X[:,1].max()+1

        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 200),
            np.linspace(y_min, y_max, 200)
        )

        grid = np.c_[xx.ravel(), yy.ravel()]
        X_design = np.c_[np.ones(grid.shape[0]), grid]

        w_map = self.get_w_map()
        z = X_design @ w_map

        sigma = np.sqrt(np.sum(X_design @ self.cov_ * X_design, axis=1))

        z_upper = z + 2 * sigma
        z_lower = z - 2 * sigma

        if ax is None:
            fig, ax = plt.subplots(figsize=(6,6))

        ax.contour(xx, yy, z.reshape(xx.shape), levels=[0], colors='black')
        ax.contour(xx, yy, z_upper.reshape(xx.shape), levels=[0], \
                   colors='red', linestyles='--')
        ax.contour(xx, yy, z_lower.reshape(xx.shape), levels=[0], \
                   colors='blue', linestyles='--')

        cmap = plt.get_cmap("bwr")
        ax.scatter(X[:,0], X[:,1], c=y, cmap=cmap, edgecolor='k')

        return ax
