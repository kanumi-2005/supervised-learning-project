import numpy as np
import matplotlib.pyplot as plt

from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan
import statsmodels.api as sm


class GaussMarkovDiagnostics:
    """
    Diagnostic tools for validating Gauss-Markov assumptions in linear models.

    This class provides visual and statistical diagnostics for assessing
    linear regression assumptions, including homoscedasticity and normality
    of residuals. It supports residual plots, QQ plots, and the
    Breusch-Pagan test for heteroscedasticity.

    Notes
    -----
    The Gauss-Markov assumptions include linearity, independence, and
    constant variance of errors. Violations can be diagnosed using the
    provided methods.

    Examples
    --------
    >>> diag = GaussMarkovDiagnostics()
    >>> diag.plot_residuals(y_true, y_pred)
    >>> diag.plot_qq(residuals)
    >>> diag.breusch_pagan_test(X, residuals)
    """

    def plot_residuals(self, y_true, y_pred, ax=None):
        """
        Plot residuals versus predicted values.

        This plot is used to visually inspect heteroscedasticity and
        model misspecification patterns.

        Parameters
        ----------
        y_true : array-like of shape (n_samples,)
            True target values.

        y_pred : array-like of shape (n_samples,)
            Predicted values from the model.

        ax : matplotlib axis, default=None
            Axis to plot on. If None, a new figure is created.

        Returns
        -------
        ax : matplotlib axis
            The axis containing the plot.
        """
        residuals = y_true - y_pred

        if ax is None:
            _, ax = plt.subplots(layout="constrained")

        ax.scatter(y_pred, residuals)
        ax.axhline(0, linestyle="--")

        ax.set_xlabel("Predicted")
        ax.set_ylabel("Residuals")
        ax.set_title("Residual Plot")
        ax.set_axisbelow(True)
        ax.grid()

        return ax

    def plot_residuals_direct(self, y_pred, residuals, ax=None):
        """
        Plot residuals directly against predictions.

        This is a lower-level version of residual plotting when residuals
        are precomputed.

        Parameters
        ----------
        y_pred : array-like of shape (n_samples,)
            Predicted values.

        residuals : array-like of shape (n_samples,)
            Precomputed residuals.

        ax : matplotlib axis, default=None
            Axis to plot on. If None, a new figure is created.

        Returns
        -------
        ax : matplotlib axis
            The axis containing the plot.
        """
        if ax is None:
            _, ax = plt.subplots(layout="constrained")

        ax.scatter(y_pred, residuals)
        ax.axhline(0, linestyle="--")

        ax.set_xlabel("Predicted")
        ax.set_ylabel("Residuals")
        ax.set_title("Residual Plot")
        ax.grid()

        return ax

    def plot_qq(self, residuals, ax=None):
        """
        Generate a QQ plot of residuals.

        The QQ plot compares empirical residual quantiles with a normal
        distribution to assess normality.

        Parameters
        ----------
        residuals : array-like of shape (n_samples,)
            Model residuals.

        ax : matplotlib axis, default=None
            Axis to plot on. If None, a new figure is created.

        Returns
        -------
        ax : matplotlib axis
            The axis containing the plot.
        """
        if ax is None:
            _, ax = plt.subplots(layout="constrained")

        stats.probplot(residuals, dist="norm", plot=ax)
        ax.set_title("QQ Plot")

        return ax

    def breusch_pagan_test(self, X, residuals):
        """
        Perform Breusch-Pagan test for heteroscedasticity.

        This test checks whether the variance of residuals depends on
        the independent variables.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Design matrix.

        residuals : array-like of shape (n_samples,)
            Model residuals.

        Returns
        -------
        dict
            Dictionary containing LM statistic, LM p-value, F statistic,
            and F p-value.
        """
        X_const = sm.add_constant(X)

        lm_stat, lm_pvalue, f_stat, f_pvalue = het_breuschpagan(
            residuals, X_const
        )

        return {
            "LM Stat": lm_stat,
            "LM p-value": lm_pvalue,
            "F Stat": f_stat,
            "F p-value": f_pvalue
        }
