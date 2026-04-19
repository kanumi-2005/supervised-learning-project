import numpy as np
from sklearn.base import RegressorMixin
from ..base.basemodel import BaseModel


class WLS(RegressorMixin, BaseModel):
    def __init__(self):
        self.intercept_ = None
        self.coef_ = None
        self.weights_ = None

    def _fit(self, X, y, **kwargs):
        X_design = np.c_[np.ones(X.shape[0]), X]
        pinv_X = np.linalg.pinv(X_design)

        w_ols = pinv_X @ y
        res_sq = np.square(y - X_design @ w_ols)

        log_res_sq = np.log(res_sq + 1e-6)
        gamma = pinv_X @ log_res_sq

        sigma2_hat = np.exp(X_design @ gamma)
        self.weights_ = 1.0 / sigma2_hat

        X_w = X_design.T * self.weights_
        A_wls = X_w @ X_design
        b_wls = X_w @ y
        w_final = np.linalg.pinv(A_wls) @ b_wls

        self.intercept_ = w_final[0]
        self.coef_ = w_final[1:]

    def _predict(self, X):
        return X @ self.coef_ + self.intercept_


if __name__ == "__main__":
    from ..dataset import CaliforniaHousingDataset as Dataset
    from sklearn.metrics import mean_squared_error

    d = Dataset()
    d.split()

    model = WLS()
    model.fit(d.X_train, d.y_train)

    y_pred = model.predict(d.X_test)
    mse = mean_squared_error(d.y_test, y_pred)

    print(f"Time: {model.training_time_:.4f}s")
    print(f"Memory: {model.training_memory_ / 1024:.2f}KB")
    print(f"MSE: {mse:.4f}")
