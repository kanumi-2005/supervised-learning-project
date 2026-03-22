import numpy as np
from sklearn.base import RegressorMixin, BaseEstimator
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF


def gradient_descent_optimizer(obj_func, initial_theta, bounds):
    theta_opt = initial_theta.copy()
    lr = 0.001
    epochs = 100

    for epoch in range(epochs):
        value, grad = obj_func(theta_opt, eval_gradient=True)
        theta_opt += lr * grad

        for i, (low, high) in enumerate(bounds):
            if low is not None:
                theta_opt[i] = max(theta_opt[i], low)
            if high is not None:
                theta_opt[i] = min(theta_opt[i], high)

    func_min = obj_func(theta_opt, eval_gradient=False)
    return theta_opt, func_min


class GPR(RegressorMixin, BaseEstimator):

    def __init__(self):
        self._gpr = GaussianProcessRegressor(
            RBF(0.2, (1e-2, 1e1)),
            optimizer=gradient_descent_optimizer,
            # n_restarts_optimizer=5,
            copy_X_train=False,
            random_state=42,
        )

    def fit(self, X, y):
        self._gpr.fit(X, y)
        self.fitted_ = True

    def predict(self, X):
        return self._gpr.predict(X, return_std=True)


if __name__ == "__main__":
    from ..dataset import CaliforniaHousingDataset as Dataset
    from ..pipeline import get_pipeline

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

    mse = np.sum(np.square(y_true - y_pred))

    print(f"MSE = {mse:.4f}")
