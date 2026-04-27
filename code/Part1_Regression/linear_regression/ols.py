import numpy as np
from sklearn.base import RegressorMixin
from ..base.basemodel import BaseModel


class OLS(RegressorMixin, BaseModel):
    """
    Ordinary Least Squares (OLS) linear regression model.

    This model estimates regression coefficients by minimizing the
    squared error between predicted and true targets using the closed-
    form solution based on the normal equation.

    The solution is computed as:
        w = (X^T X)^+ X^T y
    where (.)^+ denotes the Moore–Penrose pseudoinverse.

    The model supports an intercept term by explicitly augmenting the
    design matrix with a column of ones.

    Attributes
    ----------
    intercept_ : float
        Estimated intercept term of the linear model.

    coef_ : ndarray of shape (n_features,)
        Estimated regression coefficients for input features.

    Notes
    -----
    This implementation uses the pseudoinverse for numerical stability
    instead of direct matrix inversion.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.pipeline import Pipeline
    >>> from your_package.model import OLS
    >>> X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    >>> y = np.array([6, 8, 9, 11])
    >>> model = OLS()
    >>> model.fit(X, y)
    >>> model.predict(np.array([[3, 5]]))
    """

    def _fit(self, X, y, **kwargs):
        """
        Fit the OLS regression model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training input samples.

        y : array-like of shape (n_samples,)
            Target values.

        **kwargs : dict
            Additional keyword arguments (unused).

        Returns
        -------
        None
            The model parameters are stored in-place.
        """
        X_design = np.c_[np.ones(len(X)), X]
        A = X_design.T @ X_design
        b = X_design.T @ y
        w = np.linalg.pinv(A) @ b
        self.intercept_ = w[0]
        self.coef_ = w[1:]

    def _predict(self, X):
        """
        Predict using the learned linear model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        ndarray of shape (n_samples,)
            Predicted target values.
        """
        return X @ self.coef_ + self.intercept_


if __name__ == "__main__":
    from ..dataset import CaliforniaHousingDataset as Dataset
    from ..pipeline import get_pipeline
    from sklearn.metrics import mean_squared_error

    d = Dataset()
    d.split()

    model = get_pipeline(OLS())
    model.fit(d.X_train, d.y_train)
    y_pred = model.predict(d.X_test)
    y_true = d.y_test

    mse = mean_squared_error(y_true, y_pred)

    print(f"MSE = {mse:.4f}")
