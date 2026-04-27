import numpy as np
from sklearn.base import RegressorMixin, BaseEstimator


class IRLS(RegressorMixin, BaseEstimator):
    """
    Iteratively Reweighted Least Squares (IRLS) regression model.

    This model fits a linear regression using an iterative scheme where
    each iteration solves a weighted least squares problem. The weights
    are updated based on the residuals and a chosen robust loss function.

    The model supports robust losses such as Huber and Student-t,
    enabling resistance to outliers.

    Parameters
    ----------
    loss : {"huber", "student-t"}, default="huber"
        Loss function used to compute weights.

        - "huber": piecewise linear-quadratic loss
        - "student-t": heavy-tailed likelihood model

    delta : float, default=1.0
        Threshold parameter for Huber loss. Residuals larger than this
        are down-weighted.

    nu : float, default=4.0
        Degrees of freedom for Student-t loss.

    max_iter : int, default=50
        Maximum number of IRLS iterations.

    tol : float, default=1e-6
        Convergence tolerance for parameter updates.

    Attributes
    ----------
    beta_ : ndarray of shape (n_features + 1,)
        Estimated regression coefficients including intercept.

    intercept_ : float
        Intercept term of the model.

    coef_ : ndarray of shape (n_features,)
        Regression coefficients excluding intercept.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.random.randn(100, 2)
    >>> y = X @ np.array([1.5, -2.0]) + 0.5
    >>> model = IRLS(loss="huber")
    >>> model.fit(X, y)
    >>> model.predict(X[:3])
    array([...])
    """

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
        """
        Add intercept column to input features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        X_design : ndarray of shape (n_samples, n_features + 1)
            Data with added intercept column.
        """
        return np.c_[np.ones(len(X)), X]

    def _compute_weights(self, r):
        """
        Compute IRLS weights based on residuals.

        Parameters
        ----------
        r : ndarray of shape (n_samples,)
            Residuals.

        Returns
        -------
        w : ndarray of shape (n_samples,)
            Sample weights.
        """
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
        """
        Fit IRLS regression model.

        The algorithm alternates between computing residuals and solving
        a weighted least squares problem until convergence or reaching
        maximum iterations.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.

        y : ndarray of shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
            Fitted estimator.
        """

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
        """
        Predict using fitted IRLS model.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted values.
        """
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
