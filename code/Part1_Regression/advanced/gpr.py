import numpy as np
from sklearn.base import RegressorMixin, BaseEstimator
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF


def gradient_descent_optimizer(obj_func, initial_theta, bounds):
    theta_opt = initial_theta.copy()
    lr = 0.001
    max_iter = 100

    for _ in range(max_iter):
        value, grad = obj_func(theta_opt, eval_gradient=True)
        theta_opt += lr * grad

        for j, (low, high) in enumerate(bounds):
            if low is not None:
                theta_opt[j] = max(theta_opt[j], low)
            if high is not None:
                theta_opt[j] = min(theta_opt[j], high)

    func_min = obj_func(theta_opt, eval_gradient=False)
    return theta_opt, func_min


class GPR(RegressorMixin, BaseEstimator):

    def __init__(self, random_sate=42):
        self._gpr = GaussianProcessRegressor(
            RBF(0.2, (1e-2, 1e1)),
            optimizer=gradient_descent_optimizer,
            random_state=random_sate,
        )

    def fit(self, X, y):
        self._gpr.fit(X, y)
        self.fitted_ = True

    def predict(self, X):
        return self._gpr.predict(X, return_std=True)


if __name__ == "__main__":
    from ..dataset import CaliforniaHousingDataset as Dataset
    from ..pipeline import get_pipeline
    from sklearn.metrics import mean_squared_error

    d = Dataset()
    d.split()

    rng = np.random.default_rng(42)
    idx = rng.choice(len(d.X_train), 2000, replace=False)
    X_sub = d.X_train[idx]
    y_sub = d.y_train[idx]

    model = get_pipeline(GPR())
    model.fit(X_sub, y_sub)
    y_pred, _ = model.predict(d.X_test)
    y_true = d.y_test

    mse = mean_squared_error(y_true, y_pred)

    print(f"MSE = {mse:.4f}")
