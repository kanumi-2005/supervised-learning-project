import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error
from ..base.basegdmodel import BaseGDModel


class RidgeRegression(BaseGDModel):
    """
    Ridge Regression model trained using gradient descent.

    This model fits a linear function with L2 regularization on the
    coefficients. The objective is to minimize the mean squared error
    with an added penalty proportional to the squared magnitude of
    the coefficients.

    The optimization is performed using (stochastic) gradient descent
    depending on the batch size configuration inherited from the
    base class.

    Parameters
    ----------
    alpha : float, default=1.0
        Regularization strength. Must be a non-negative float.
        Larger values specify stronger regularization.

    lr : float, default=0.001
        Learning rate used in gradient descent updates.

    max_iter : int, default=1000
        Maximum number of iterations for the optimization process.

    store_history : bool, default=False
        Whether to store loss values during training.

    batch_size : int or None, default=None
        Number of samples per batch. If None, full batch gradient
        descent is used.

    warm_start : bool, default=False
        If True, reuse the solution of the previous call to fit
        as initialization.

    random_state : int, default=42
        Seed used for random number generation.

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        Estimated coefficients for the linear model.

    intercept_ : float
        Independent term in the linear model.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    >>> y = np.dot(X, np.array([1.0, 2.0])) + 3.0
    >>> model = RidgeRegression(alpha=1.0, lr=0.01, max_iter=1000)
    >>> model.fit(X, y)
    >>> model.coef_
    array([1., 2.])
    >>> model.intercept_
    3.0
    >>> model.predict(np.array([[3, 5]]))
    array([16.])
    """

    def __init__(
        self,
        alpha=1.0,
        lr=0.001,
        max_iter=1000,
        store_history=False,
        batch_size=None,
        warm_start=False,
        random_state=42
    ):
        """
        Initialize the RidgeRegression model.

        Parameters
        ----------
        alpha : float, default=1.0
            Regularization strength.

        lr : float, default=0.001
            Learning rate.

        max_iter : int, default=1000
            Maximum number of iterations.

        store_history : bool, default=False
            Whether to store training loss history.

        batch_size : int or None, default=None
            Size of mini-batches.

        warm_start : bool, default=False
            Reuse previous solution if available.

        random_state : int, default=42
            Random seed.
        """
        self.alpha = alpha
        self.warm_start = warm_start
        super().__init__(
            lr=lr,
            max_iter=max_iter,
            store_history=store_history,
            batch_size=batch_size,
            random_state=random_state
        )

    def _predict(self, X):
        """
        Compute predictions using the linear model.

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

    def _init_params(self, X, y):
        """
        Initialize model parameters.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.

        y : ndarray of shape (n_samples,)
            Target values.

        Notes
        -----
        If warm_start is enabled and parameters already exist,
        initialization is skipped.
        """
        n_features = X.shape[1]

        if self.warm_start and hasattr(self, "coef_"):
            return

        self.coef_ = np.zeros(n_features)
        self.intercept_ = 0.0

    def _loss(self, X, y):
        """
        Compute the Ridge loss.

        The loss consists of mean squared error plus L2 penalty.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data.

        y : ndarray of shape (n_samples,)
            True target values.

        Returns
        -------
        loss : float
            Computed loss value.
        """
        y_pred = self._predict(X)
        mse = np.mean((y_pred - y) ** 2)
        reg = self.alpha * np.sum(self.coef_ ** 2)
        return mse + reg

    def _grad(self, X, y):
        """
        Compute gradients of the loss function.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data.

        y : ndarray of shape (n_samples,)
            True target values.

        Returns
        -------
        grad_w : ndarray of shape (n_features,)
            Gradient with respect to coefficients.

        grad_b : float
            Gradient with respect to intercept.
        """
        n_samples = X.shape[0]

        y_pred = self._predict(X)
        errors = y_pred - y

        grad_w = (2.0 / n_samples) * (X.T @ errors) + \
            2 * self.alpha * self.coef_
        grad_b = (2.0 / n_samples) * np.sum(errors)

        return grad_w, grad_b

    def _update_params(self, grad, iteration):
        """
        Update model parameters using gradient descent.

        Parameters
        ----------
        grad : tuple
            Tuple containing gradients (grad_w, grad_b).

        iteration : int
            Current iteration number.

        Returns
        -------
        None
        """
        grad_w, grad_b = grad

        self.coef_ -= self.lr * grad_w
        self.intercept_ -= self.lr * grad_b


class RidgeRegressionCV(BaseEstimator, RegressorMixin):
    """
    Ridge Regression with built-in cross-validation.

    This estimator selects the optimal regularization parameter
    from a predefined set of alphas using K-fold cross-validation.
    For each alpha, the model is trained and evaluated across
    multiple folds, and the alpha with the lowest average mean
    squared error is selected.

    Parameters
    ----------
    alphas : iterable of float, default=(0.1, 1.0, 10.0)
        List of regularization strengths to evaluate.

    cv : int, default=10
        Number of cross-validation folds.

    random_state : int, default=42
        Seed used for shuffling data in cross-validation.

    Attributes
    ----------
    best_alpha_ : float
        Selected regularization parameter with lowest CV error.

    coef_ : ndarray of shape (n_features,)
        Coefficients of the final fitted model.

    intercept_ : float
        Intercept of the final fitted model.

    result_ : ndarray of shape (n_alphas,)
        Mean squared error for each alpha averaged over folds.

    coefs_path_ : ndarray of shape (n_alphas, n_features)
        Average coefficient values across folds for each alpha.

    final_model_ : RidgeRegression
        Model fitted on the full dataset using best_alpha_.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    >>> y = np.dot(X, np.array([1.0, 2.0])) + 3.0
    >>> model = RidgeRegressionCV(alphas=(0.1, 1.0, 10.0), cv=2)
    >>> model.fit(X, y)
    >>> model.best_alpha_
    1.0
    >>> model.coef_
    array([1., 2.])
    >>> model.predict(np.array([[3, 5]]))
    array([16.])
    """

    def __init__(self, alphas=(0.1, 1.0, 10.0), cv=10, random_state=42):
        """
        Initialize the RidgeRegressionCV estimator.

        Parameters
        ----------
        alphas : iterable of float, default=(0.1, 1.0, 10.0)
            Candidate regularization strengths.

        cv : int, default=10
            Number of folds for cross-validation.

        random_state : int, default=42
            Random seed for reproducibility.
        """
        self.alphas = alphas
        self.cv = cv
        self.random_state = random_state

    def fit(self, X, y):
        """
        Fit Ridge Regression model with cross-validation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
            Fitted estimator with selected alpha.
        """
        X, y = np.array(X), np.array(y).ravel()
        self.alphas = sorted(self.alphas, reverse=True)
        kf = KFold(
            n_splits=self.cv,
            shuffle=True,
            random_state=self.random_state
        )
        mse_scores = np.zeros(len(self.alphas))
        coefs_folds = []

        for train_idx, val_idx in kf.split(X):
            fold_model = RidgeRegression(
                warm_start=True,
            )
            fold_coefs = []

            for i, alpha in enumerate(self.alphas):
                fold_model.set_params(alpha=alpha)
                fold_model.fit(X[train_idx], y[train_idx])

                pred = fold_model.predict(X[val_idx])
                mse_scores[i] += mean_squared_error(y[val_idx], pred)

                fold_coefs.append(fold_model.coef_.copy())

            coefs_folds.append(fold_coefs)

        avg_mse = mse_scores / self.cv
        best_idx = np.argmin(avg_mse)
        self.best_alpha_ = self.alphas[best_idx]
        self.result_ = avg_mse

        coefs_folds = np.array(coefs_folds)
        self.coefs_path_ = np.mean(coefs_folds, axis=0)

        self.final_model_ = RidgeRegression(
            alpha=self.best_alpha_,
            warm_start=False
        )

        self.final_model_.fit(X, y)
        self.coef_ = self.final_model_.coef_
        self.intercept_ = self.final_model_.intercept_

        return self

    def predict(self, X):
        """
        Predict using the fitted Ridge model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted values.
        """
        return self.final_model_.predict(X)

    def plot_regularization_path(self, title=None):
        """
        Plot coefficient paths over different alphas.

        The plot shows how each feature coefficient changes
        as a function of the logarithm of the regularization
        parameter.

        Parameters
        ----------
        title : str or None, default=None
            Title of the plot.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the model has not been fitted yet.
        """
        if self.coefs_path_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        alphas = np.array(self.alphas)
        coefs = self.coefs_path_

        fig, ax = plt.subplots(layout="constrained")

        for i in range(coefs.shape[1]):
            ax.plot(np.log10(alphas), coefs[:, i], label=f'Feature {i+1}')

        ax.set_title(title)
        ax.set_xlabel(r'$\log_{10}(\lambda)$')
        ax.set_ylabel('Coefficients')
        ax.grid(True)
        ax.legend(loc='best', fontsize='small')

        plt.show()
