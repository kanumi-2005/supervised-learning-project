from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV


class KernelRidgeCV(BaseEstimator, RegressorMixin):
    """
    Kernel Ridge regression with built-in cross-validation.

    This estimator performs hyperparameter tuning for Kernel Ridge
    regression using cross-validation over a specified parameter grid.
    It supports radial basis function (RBF) and polynomial kernels.

    Parameters
    ----------
    kernel : {'rbf', 'polynomial'}, default='rbf'
        Kernel type to be used in the model.

    param_grid : dict, default=None
        Dictionary with parameters names (`str`) as keys and lists of
        parameter settings to try as values. If None, a default grid
        is used depending on the kernel.

    cv : int, default=10
        Number of folds in cross-validation.

    Attributes
    ----------
    model_ : estimator instance
        The best KernelRidge model found during grid search.

    best_params_ : dict
        Parameter setting that gave the best results.

    best_score_ : float
        Mean cross-validated score of the best_estimator.

    grid_ : GridSearchCV instance
        The underlying GridSearchCV object.

    Notes
    -----
    This implementation is a wrapper around GridSearchCV applied to
    Kernel Ridge regression.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.datasets import make_regression
    >>> X, y = make_regression(n_samples=100, n_features=5,
    ...                        noise=0.1, random_state=42)
    >>> model = KernelRidgeCV(kernel='rbf')
    >>> model.fit(X, y)
    KernelRidgeCV(...)
    >>> preds = model.predict(X)
    >>> model.score(X, y)
    1.0
    """

    def __init__(self, kernel='rbf', param_grid=None, cv=10):
        """
        Initialize the KernelRidgeCV estimator.

        Parameters
        ----------
        kernel : {'rbf', 'polynomial'}, default='rbf'
            Kernel type to be used.

        param_grid : dict, default=None
            Grid of hyperparameters to search.

        cv : int, default=10
            Number of cross-validation folds.
        """
        self.kernel = kernel
        self.param_grid = param_grid
        self.cv = cv

    def _default_param_grid(self):
        """
        Return default parameter grid based on kernel type.

        Returns
        -------
        param_grid : dict
            Default grid for hyperparameter search.

        Raises
        ------
        ValueError
            If the kernel is not supported.
        """
        if self.kernel == 'rbf':
            return {
                "gamma": [0.01, 0.1, 1]
            }
        elif self.kernel == 'polynomial':
            return {
                "degree": [2, 3],
                "coef0": [0, 1]
            }
        else:
            raise ValueError("Kernel must be 'rbf' or 'polynomial'")

    def fit(self, X, y):
        """
        Fit the Kernel Ridge model using grid search.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        param_grid = self.param_grid if self.param_grid is not None \
            else self._default_param_grid()

        self.grid_ = GridSearchCV(
            KernelRidge(kernel=self.kernel),
            param_grid,
            cv=self.cv
        )

        self.grid_.fit(X, y)

        self.model_ = self.grid_.best_estimator_
        self.best_params_ = self.grid_.best_params_
        self.best_score_ = self.grid_.best_score_

        return self

    def predict(self, X):
        """
        Predict using the best found model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        y_pred : array of shape (n_samples,)
            Predicted values.
        """
        return self.model_.predict(X)

    def score(self, X, y):
        """
        Return the coefficient of determination R^2 of the prediction.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        y : array-like of shape (n_samples,)
            True values for X.

        Returns
        -------
        score : float
            R^2 score of the prediction.
        """
        return self.model_.score(X, y)
