import numpy as np
from sklearn.base import RegressorMixin, BaseEstimator
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF, WhiteKernel, ConstantKernel
)


def gradient_descent_optimizer(obj_func, initial_theta, bounds):
    """
    Simple gradient descent optimizer for hyperparameter tuning.

    This optimizer performs a fixed number of gradient descent steps
    to minimize the provided objective function, while enforcing box
    constraints on parameters.

    Parameters
    ----------
    obj_func : callable
        Objective function with signature:
        obj_func(theta, eval_gradient=True/False).
        It must return (value, gradient) when eval_gradient=True.

    initial_theta : array-like of shape (n_params,)
        Initial parameter vector.

    bounds : list of tuple
        Bounds for each parameter as (lower, upper). Use None for
        unbounded sides.

    Returns
    -------
    theta_opt : ndarray of shape (n_params,)
        Optimized parameters.

    func_min : float
        Final objective function value.
    """
    theta_opt = initial_theta.copy()
    lr = 0.01

    for _ in range(200):
        value, grad = obj_func(theta_opt, eval_gradient=True)

        theta_opt -= lr * grad

        for j, (low, high) in enumerate(bounds):
            if low is not None:
                theta_opt[j] = max(theta_opt[j], low)
            if high is not None:
                theta_opt[j] = min(theta_opt[j], high)

    func_min = obj_func(theta_opt, eval_gradient=False)
    return theta_opt, func_min


class GPR(RegressorMixin, BaseEstimator):
    """
    Gaussian Process Regression with custom optimizer.

    This model wraps sklearn's GaussianProcessRegressor and replaces
    its internal optimizer with a custom gradient descent procedure.

    The model assumes a kernel composed of:
    ConstantKernel * RBF + WhiteKernel.

    Parameters
    ----------
    alpha : float, default=1e-2
        Value added to the diagonal of the kernel matrix during fitting
        for numerical stability.

    random_state : int, default=42
        Controls randomness of the underlying Gaussian process.

    Attributes
    ----------
    _gpr : GaussianProcessRegressor
        Internal sklearn Gaussian process model.

    fitted_ : bool
        Whether the model has been fitted.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.random.rand(10, 2)
    >>> y = np.sin(X[:, 0])
    >>> model = GPR()
    >>> model.fit(X, y)
    >>> mean, std = model.predict(X)
    """

    def __init__(self, alpha=1e-2, random_state=42):
        self.random_state = random_state
        self._gpr = GaussianProcessRegressor(
            ConstantKernel() * RBF() + WhiteKernel(),
            alpha=alpha,
            optimizer=gradient_descent_optimizer,
            random_state=random_state,
        )

    def fit(self, X, y):
        """
        Fit the Gaussian process regression model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training input data.

        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        self._gpr.fit(X, y)
        self.fitted_ = True

    def predict(self, X):
        """
        Predict using the Gaussian process model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        mean : ndarray of shape (n_samples,)
            Predicted mean values.

        std : ndarray of shape (n_samples,)
            Predicted standard deviation.
        """
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
