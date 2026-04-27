import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error
from ..base.basegdmodel import BaseGDModel


class ElasticNet(BaseGDModel):
    """
    Linear regression with combined L1 and L2 regularization.

    ElasticNet fits a linear model with coefficients w = (w1, ..., wp)
    by minimizing the mean squared error with both L1 (lasso) and
    L2 (ridge) penalties. This combination encourages sparsity while
    maintaining stability of the solution.

    Parameters
    ----------
    alpha_1 : float, default=1.0
        Constant that multiplies the L1 penalty term.

    alpha_2 : float, default=0.5
        Constant that multiplies the L2 penalty term.

    lr : float, default=0.001
        Learning rate for gradient descent optimization.

    max_iter : int, default=1000
        Maximum number of iterations for the optimization.

    store_history : bool, default=False
        Whether to store loss history during training.

    batch_size : int, default=None
        Size of mini-batches for stochastic gradient descent.
        If None, full batch gradient descent is used.

    random_state : int, default=42
        Seed for random number generation.

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        Estimated coefficients for the regression problem.

    intercept_ : float
        Independent term in the linear model.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    >>> y = np.dot(X, np.array([1, 2])) + 3
    >>> model = ElasticNet(alpha_1=0.1, alpha_2=0.1)
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
        alpha_1=1.0,
        alpha_2=0.5,
        lr=0.001,
        max_iter=1000,
        store_history=False,
        batch_size=None,
        random_state=42
    ):
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        super().__init__(
            lr=lr,
            max_iter=max_iter,
            store_history=store_history,
            batch_size=batch_size,
            random_state=random_state
        )

    def _predict(self, X):
        """
        Predict using the linear model.

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
        """
        n_features = X.shape[1]
        self.coef_ = np.zeros(n_features)
        self.intercept_ = 0.0

    def _loss(self, X, y):
        """
        Compute the loss function.

        The loss is the sum of mean squared error and L1 and L2 penalties.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data.

        y : ndarray of shape (n_samples,)
            Target values.

        Returns
        -------
        loss : float
            Computed loss value.
        """
        y_pred = self._predict(X)
        mse = np.mean((y_pred - y) ** 2)
        l1 = self.alpha_1 * np.sum(np.abs(self.coef_))
        l2 = self.alpha_2 * np.sum(self.coef_ ** 2)
        return mse + l1 + l2

    def _grad(self, X, y):
        """
        Compute gradients of the loss with respect to parameters.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data.

        y : ndarray of shape (n_samples,)
            Target values.

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

        grad_mse = (2.0 / n_samples) * (X.T @ errors)
        grad_l2 = 2.0 * self.alpha_2 * self.coef_
        grad_l1 = self.alpha_1 * np.sign(self.coef_)

        grad_w = grad_mse + grad_l2 + grad_l1
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
        """
        grad_w, grad_b = grad

        self.coef_ -= self.lr * grad_w
        self.intercept_ -= self.lr * grad_b


class ElasticNetCV(BaseEstimator, RegressorMixin):
    """
    Elastic Net model with cross-validation.

    This estimator selects optimal values for L1 and L2 regularization
    parameters using K-fold cross-validation. The best combination of
    (alpha_1, alpha_2) is chosen based on mean squared error.

    Parameters
    ----------
    alpha_1s : iterable of float, default=(0.1, 1.0, 10.0)
        Candidate values for the L1 regularization parameter.

    alpha_2s : iterable of float, default=(0.1, 1.0, 10.0)
        Candidate values for the L2 regularization parameter.

    cv : int, default=10
        Number of folds in K-fold cross-validation.

    random_state : int, default=42
        Seed for reproducibility when shuffling data.

    Attributes
    ----------
    best_alpha_1_ : float
        Optimal value of the L1 regularization parameter.

    best_alpha_2_ : float
        Optimal value of the L2 regularization parameter.

    results_ : ndarray of shape (len(alpha_2s), len(alpha_1s))
        Mean squared error for each parameter combination.

    final_model_ : ElasticNet
        Model trained on the full dataset with optimal parameters.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    >>> y = np.dot(X, np.array([1, 2])) + 3
    >>> model = ElasticNetCV()
    >>> model.fit(X, y)
    >>> model.best_alpha_1_
    0.1
    >>> model.best_alpha_2_
    0.1
    >>> model.predict(np.array([[3, 5]]))
    array([16.])
    """

    def __init__(
            self,
            alpha_1s=(0.1, 1.0, 10.0),
            alpha_2s=(0.1, 1.0, 10.0),
            cv=10,
            random_state=42
        ):

        self.alpha_1s = alpha_1s
        self.alpha_2s = alpha_2s
        self.cv = cv
        self.random_state = random_state

    def fit(self, X, y):
        """
        Fit Elastic Net model with cross-validation.

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
        X, y = np.array(X), np.array(y).ravel()
        self.alpha_1s = sorted(self.alpha_1s, reverse=True)
        self.alpha_2s = sorted(self.alpha_2s, reverse=True)
        kf = KFold(
            n_splits=self.cv,
            shuffle=True,
            random_state=self.random_state
        )

        alpha_1_grid, alpha_2_grid = np.meshgrid(self.alpha_1s, self.alpha_2s)
        param_grid = np.vstack([alpha_1_grid.ravel(), alpha_2_grid.ravel()]).T
        mse_results = np.zeros(len(param_grid))

        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            fold_model = ElasticNet()

            for i, (alpha_1, alpha_2) in enumerate(param_grid):
                fold_model.set_params(alpha_1=alpha_1, alpha_2=alpha_2)
                fold_model.fit(X_train, y_train)

                pred = fold_model.predict(X_val)
                mse_results[i] += mean_squared_error(y_val, pred)

        avg_mse = mse_results / self.cv
        best_idx = np.argmin(avg_mse)
        self.best_alpha_1_, self.best_alpha_2_ = param_grid[best_idx]

        self.results_ = avg_mse.reshape(
            len(self.alpha_2s),
            len(self.alpha_1s)
        )

        self.final_model_ = ElasticNet(
            alpha_1=self.best_alpha_1_,
            alpha_2=self.best_alpha_2_
        )

        self.final_model_.fit(X, y)

        return self

    def predict(self, X):
        """
        Predict using the fitted Elastic Net model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted values.
        """
        return self.final_model_.predict(X)

    def plot_optimal_region(self, title=None):
        """
        Plot contour of mean squared error over parameter grid.

        Parameters
        ----------
        title : str, default=None
            Title for the plot.
        """
        plot_alpha_1s = np.sort(self.alpha_1s)
        plot_alpha_2s = np.sort(self.alpha_2s)

        log_alpha_1, log_alpha_2 = np.meshgrid(
            np.log10(plot_alpha_1s),
            np.log10(plot_alpha_2s)
        )

        flipped_results = np.flip(self.results_, axis=(0, 1))

        fig, ax = plt.subplots(layout="constrained")

        cp = ax.contourf(
            log_alpha_1,
            log_alpha_2,
            flipped_results,
            levels=20,
            cmap='viridis_r'
        )

        ax.plot(
            np.log10(self.best_alpha_1_),
            np.log10(self.best_alpha_2_),
            'ro',
            markersize=15,
            markeredgecolor='black',
            label='Optimal Point'
        )

        fig.colorbar(cp, ax=ax, label='Mean MSE')

        ax.set_xlabel(r'$\log_{10}(\lambda_1)$')
        ax.set_ylabel(r'$\log_{10}(\lambda_2)$')
        ax.set_title(title)
        ax.legend()

        plt.show()
