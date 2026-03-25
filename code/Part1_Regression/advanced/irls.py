import numpy as np
from sklearn.base import RegressorMixin, BaseEstimator


class IRLS(RegressorMixin, BaseEstimator):
    def __init__(
        self,
        loss="huber",
        delta=1.0,
        nu=4.0,
        max_iter=50,
        tol=1e-6,
    ):
        self.loss = loss
        self.delta = delta
        self.nu = nu
        self.max_iter = max_iter
        self.tol = tol

    def _add_intercept(self, X):
        return np.c_[np.ones(len(X)), X]

    def _compute_weights(self, r):
        eps = 1e-8
        if self.loss == "huber":
            abs_r = np.abs(r)
            w = np.ones_like(r)
            mask = abs_r > self.delta
            w[mask] = self.delta / (abs_r[mask] + eps)
            return w
        elif self.loss == "student-t":
            return (self.nu + 1) / (self.nu + r**2)
        else:
            raise ValueError

    def fit(self, X, y):

        X_design = self._add_intercept(X)

        beta = np.linalg.lstsq(X_design, y, rcond=None)[0]

        for _ in range(self.max_iter):
            r = y - X_design @ beta
            w = self._compute_weights(r)

            WX = X_design * w[:, None]
            A = X_design.T @ WX
            b = X_design.T @ (w * y)

            beta_new = np.linalg.solve(A, b)

            if np.linalg.norm(beta_new - beta) < self.tol:
                break

            beta = beta_new

        self.beta_ = beta
        self.intercept_ = beta[0]
        self.coef_ = beta[1:]

        return self

    def predict(self, X):
        return X @ self.coef_ + self.intercept_


if __name__ == "__main__":
    from ..dataset import CaliforniaHousingDataset as Dataset
    from ..pipeline import get_pipeline
    from sklearn.metrics import mean_squared_error

    d = Dataset()
    d.split()

    model = get_pipeline(IRLS(loss="student-t"))
    model.fit(d.X_train, d.y_train)
    y_pred = model.predict(d.X_test)
    y_true = d.y_test

    mse = mean_squared_error(y_true, y_pred)

    print(f"MSE = {mse:.4f}")
