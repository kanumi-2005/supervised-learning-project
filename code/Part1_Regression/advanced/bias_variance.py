import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, clone
from sklearn.utils import resample, check_random_state


class BiasVarianceAnalyzer(BaseEstimator):
    """
    Bias-Variance decomposition using bootstrap resampling.

    This class estimates the bias^2 and variance of a given estimator
    across a range of regularization strengths using bootstrap sampling.

    Parameters
    ----------
    estimator : estimator object
        The base estimator to evaluate. It must implement fit and predict.

    n_bootstrap : int, default=200
        Number of bootstrap samples used to estimate bias and variance.

    random_state : int, RandomState instance or None, default=42
        Controls the randomness of the bootstrap sampling.

    Attributes
    ----------
    lambdas_ : ndarray of shape (n_lambdas,)
        Regularization parameters used in the analysis.

    bias2_ : ndarray of shape (n_lambdas,)
        Estimated squared bias for each lambda.

    variance_ : ndarray of shape (n_lambdas,)
        Estimated variance for each lambda.

    X_ : array-like of shape (n_samples, n_features)
        Training data used in fit.

    y_ : array-like of shape (n_samples,)
        Target values used in fit.

    Notes
    -----
    This implementation uses bootstrap resampling to approximate
    the bias-variance decomposition.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import Ridge
    >>> X = np.random.randn(100, 5)
    >>> y = X @ np.array([1, 2, 3, 4, 5]) + np.random.randn(100)
    >>> analyzer = BiasVarianceAnalyzer(Ridge())
    >>> analyzer.fit(X, y)
    BiasVarianceAnalyzer(...)
    >>> bias2, var = analyzer.get_bias_variance()
    >>> fig, ax = analyzer.plot()
    """

    def __init__(self, estimator, n_bootstrap=200, random_state=42):
        """
        Initialize the BiasVarianceAnalyzer.

        Parameters
        ----------
        estimator : estimator object
            Base estimator implementing fit and predict.

        n_bootstrap : int, default=200
            Number of bootstrap samples.

        random_state : int, RandomState instance or None, default=42
            Random seed for reproducibility.
        """
        self.estimator = estimator
        self.n_bootstrap = n_bootstrap
        self.random_state = random_state

    def fit(self, X, y, lambdas=None):
        """
        Estimate bias^2 and variance across regularization strengths.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,)
            Target values.

        lambdas : array-like of shape (n_lambdas,), default=None
            Regularization strengths. If None, a default log-spaced
            grid is used.

        Returns
        -------
        self : object
            Fitted analyzer.
        """
        self.X_ = X
        self.y_ = y

        self.lambdas_ = (
            np.logspace(-10, 1, 12)
            if lambdas is None
            else np.asarray(lambdas)
        )

        rng = check_random_state(self.random_state)

        self.bias2_ = []
        self.variance_ = []

        for l in self.lambdas_:
            model = clone(self.estimator)
            if hasattr(model, "alpha"):
                model.alpha = l

            predictions = []

            for _ in range(self.n_bootstrap):
                X_s, y_s = resample(X, y, random_state=rng)
                m = clone(model)
                m.fit(X_s, y_s)
                predictions.append(m.predict(X))

            predictions = np.array(predictions)

            mean_pred = np.mean(predictions, axis=0)

            bias2 = np.mean((mean_pred - y) ** 2)
            variance = np.mean(np.var(predictions, axis=0, ddof=1))

            self.bias2_.append(bias2)
            self.variance_.append(variance)

        self.bias2_ = np.array(self.bias2_)
        self.variance_ = np.array(self.variance_)

        return self

    def plot(self):
        """
        Plot bias^2 and variance as a function of lambda.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The created figure.

        ax : matplotlib.axes.Axes
            The axes of the plot.
        """
        fig, ax = plt.subplots(layout="constrained")

        ax.plot(self.lambdas_, self.bias2_, label="Bias²")
        ax.plot(self.lambdas_, self.variance_, label="Variance")

        ax.set_xscale("log")
        ax.set_xlabel(r"$log_{10}(\lambda)$")
        ax.set_ylabel("Error")
        ax.set_title("Bias–Variance vs Regularization Path")
        ax.legend(loc="best")

        return fig, ax

    def get_bias_variance(self):
        """
        Return computed bias^2 and variance.

        Returns
        -------
        bias2 : ndarray of shape (n_lambdas,)
            Squared bias values.

        variance : ndarray of shape (n_lambdas,)
            Variance values.
        """
        return self.bias2_, self.variance_
