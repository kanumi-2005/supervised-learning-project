import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    """
    Numerically stable sigmoid function.

    Parameters
    ----------
    x : array-like
        Input values.

    Returns
    -------
    out : ndarray
        Sigmoid of input.
    """
    x = np.clip(x, -50, 50)
    return 1 / (1 + np.exp(-x))


class LaplaceApprox:
    """
    Laplace approximation for Bayesian logistic regression models.

    This class approximates the posterior of a trained logistic model
    using a Gaussian centered at the MAP estimate. The covariance is
    computed from the Hessian of the negative log-likelihood.

    Parameters
    ----------
    model : object
        Trained logistic-like model with intercept_ and coef_.

    reg_lambda : float, default=1e-4
        L2 regularization strength used in Hessian computation.

    Attributes
    ----------
    H : ndarray
        Hessian matrix at MAP estimate.

    cov_ : ndarray
        Approximated covariance matrix of parameters.

    Notes
    -----
    The Laplace approximation assumes a Gaussian posterior around
    the MAP solution using a second-order Taylor expansion.

    Examples
    --------
    >>> approx = LaplaceApprox(model)
    >>> approx.fit(X_train)
    >>> approx.cov_.shape
    (d, d)
    """

    def __init__(self, model, reg_lambda=1e-4):
        self.model = model
        self.reg_lambda = reg_lambda

    def get_w_map(self):
        """
        Get MAP parameter vector from trained model.

        Returns
        -------
        w : ndarray
            Concatenated intercept and coefficients.
        """
        return np.r_[self.model.intercept_, self.model.coef_.ravel()]

    def compute_hessian(self, X):
        """
        Compute Hessian of negative log-likelihood.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input features.

        Returns
        -------
        H : ndarray
            Hessian matrix.

        X_design : ndarray
            Design matrix with bias term.
        """
        n = X.shape[0]
        X_design = np.c_[np.ones(n), X]

        w_map = self.get_w_map()
        z = X_design @ w_map
        p = sigmoid(z)

        W_diag_ = np.clip(p * (1 - p), 1e-8, None)
        H = X_design.T @ (W_diag_[:, None] * X_design)
        H += self.reg_lambda * np.eye(X_design.shape[1])

        return H, X_design

    def fit(self, X):
        """
        Fit Laplace approximation by computing covariance.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data used for Hessian estimation.

        Returns
        -------
        self : LaplaceApprox
            Fitted instance.
        """
        self.H, _ = self.compute_hessian(X)
        self.cov_ = np.linalg.inv(self.H)
        return self

    def decision_boundary_sigma(self, X, y, ax=None):
        """
        Plot decision boundary with uncertainty bands.

        Parameters
        ----------
        X : ndarray of shape (n_samples, 2)
            Input features.

        y : ndarray of shape (n_samples,)
            Class labels.

        ax : matplotlib axis, optional
            Axis to plot on.

        Returns
        -------
        ax : matplotlib axis
            Axis with plotted decision boundaries.
        """
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 200),
            np.linspace(y_min, y_max, 200)
        )

        grid = np.c_[xx.ravel(), yy.ravel()]
        X_design = np.c_[np.ones(grid.shape[0]), grid]

        w_map = self.get_w_map()
        z = X_design @ w_map

        sigma = np.sqrt(
            np.einsum("ij,jk,ik->i", X_design, self.cov_, X_design)
        )

        z_upper = z + 2 * sigma
        z_lower = z - 2 * sigma

        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))

        ax.contour(xx, yy, z.reshape(xx.shape), levels=[0],
                   colors='black')
        ax.contour(xx, yy, z_upper.reshape(xx.shape), levels=[0],
                   colors='red', linestyles='--')
        ax.contour(xx, yy, z_lower.reshape(xx.shape), levels=[0],
                   colors='blue', linestyles='--')

        cmap = plt.get_cmap("bwr")
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, edgecolor='k')

        return ax

    def decision_boundary_sigma_2features(self, X, y, f1=0, f2=1,
                                         ax=None):
        """
        Plot decision boundary for two selected features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input features.

        y : ndarray of shape (n_samples,)
            Class labels.

        f1 : int, default=0
            First feature index.

        f2 : int, default=1
            Second feature index.

        ax : matplotlib axis, optional
            Axis to plot on.

        Returns
        -------
        ax : matplotlib axis
            Axis with plotted decision boundaries.
        """
        n, d = X.shape

        x_min, x_max = X[:, f1].min() - 1, X[:, f1].max() + 1
        y_min, y_max = X[:, f2].min() - 1, X[:, f2].max() + 1

        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 200),
            np.linspace(y_min, y_max, 200)
        )

        X_mean = np.mean(X, axis=0)
        grid = np.tile(X_mean, (xx.size, 1))
        grid[:, f1] = xx.ravel()
        grid[:, f2] = yy.ravel()

        X_design = np.c_[np.ones(grid.shape[0]), grid]

        w_map = self.get_w_map()
        z = X_design @ w_map

        sigma = np.sqrt(
            np.einsum("ij,jk,ik->i", X_design, self.cov_, X_design)
        )

        z_upper = z + 2 * sigma
        z_lower = z - 2 * sigma

        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))

        ax.contour(xx, yy, z.reshape(xx.shape), levels=[0],
                   colors='black')
        ax.contour(xx, yy, z_upper.reshape(xx.shape), levels=[0],
                   colors='red', linestyles='--')
        ax.contour(xx, yy, z_lower.reshape(xx.shape), levels=[0],
                   colors='blue', linestyles='--')

        cmap = plt.get_cmap("bwr")
        ax.scatter(X[:, f1], X[:, f2], c=y, cmap=cmap, edgecolor='k')

        ax.set_xlabel(f"Feature {f1}")
        ax.set_ylabel(f"Feature {f2}")
        ax.set_title("Decision Boundary ±2σ (Laplace)")

        return ax
