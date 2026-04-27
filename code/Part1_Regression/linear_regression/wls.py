import numpy as np
from sklearn.base import RegressorMixin
from ..base.basemodel import BaseModel


class WLS(RegressorMixin, BaseModel):
    """
    Weighted Least Squares (WLS) regression model.

    This model extends ordinary least squares by assigning observation
    weights inversely proportional to an estimated error variance. The
    procedure first estimates residual variance using an OLS fit, then
    re-weights samples and refits the linear model.

    The final model minimizes a weighted sum of squared residuals:

        sum_i w_i (y_i - X_i w)^2

    where weights w_i are estimated from residual variance.

    Parameters
    ----------
    None

    Attributes
    ----------
    intercept_ : float
        Intercept term of the fitted model.

    coef_ : ndarray of shape (n_features,)
        Regression coefficients of the fitted model.

    weights_ : ndarray of shape (n_samples,)
        Learned sample weights based on residual variance.

    Notes
    -----
    The method uses a two-stage estimation:
    1. Ordinary Least Squares to estimate residuals.
    2. Log-variance regression to estimate heteroscedasticity.
    3. Final weighted least squares solution.

    Examples
    --------
    >>> from your_module import WLS
    >>> import numpy as np
    >>> X = np.array([[1], [2], [3], [4]])
    >>> y = np.array([2, 4, 6, 8])
    >>> model = WLS()
    >>> model.fit(X, y)
    >>> model.predict(X)
    array([...])
    """

    def __init__(self):
        """
        Initialize WLS model.

        Initializes model parameters to None. They are learned during fit.

        Parameters
        ----------
        None
        """
        self.intercept_ = None
        self.coef_ = None
        self.weights_ = None

    def _fit(self, X, y, **kwargs):
        """
        Fit Weighted Least Squares regression model.

        The fitting procedure is performed in three steps:
        1. Compute OLS solution and residuals.
        2. Estimate log-variance via pseudo-inverse regression.
        3. Compute weighted least squares solution.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.

        y : ndarray of shape (n_samples,)
            Target values.

        **kwargs : dict
            Additional arguments (unused).

        Returns
        -------
        None
        """
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
        """
        Predict using the fitted WLS model.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted values.
        """
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
