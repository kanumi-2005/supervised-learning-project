import numpy as np
import matplotlib.pyplot as plt

from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan
import statsmodels.api as sm


class GaussMarkovDiagnostics:
    def plot_residuals(self, y_true, y_pred, ax=None):
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
        if ax is None:
            _, ax = plt.subplots(layout="constrained")

        stats.probplot(residuals, dist="norm", plot=ax)
        ax.set_title("QQ Plot")

        return ax

    def breusch_pagan_test(self, X, residuals):
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


if __name__ == "__main__":
    from ..dataset import CaliforniaHousingDataset as Dataset
    from ..pipeline import get_pipeline
    from sklearn.metrics import mean_squared_error
    from .ols import OLS
    from .wls import WLS

    d = Dataset()
    d.split()

    model = get_pipeline(OLS())
    model.fit(d.X_train, d.y_train)

    y_pred_train = model.predict(d.X_train)
    residuals_train = d.y_train - y_pred_train

    diag = GaussMarkovDiagnostics()

    diag.plot_residuals(d.y_train, y_pred_train)
    diag.plot_qq(residuals_train)

    result = diag.breusch_pagan_test(d.X_train, residuals_train)
    print(result)

    plt.show()

    if result["LM p-value"] < 0.05:
        aux = get_pipeline(OLS())
        aux.fit(d.X_train, np.square(residuals_train))

        sigma2_hat = aux.predict(d.X_train)
        sigma2_hat = np.clip(sigma2_hat, 1e-6, None)

        weights = 1.0 / sigma2_hat

        wls = get_pipeline(WLS(weights=weights))
        wls.fit(d.X_train, d.y_train)

        y_pred_train = wls.predict(d.X_train)
        residuals_train = d.y_train - y_pred_train
        weighted_residuals = np.sqrt(weights) * residuals_train

        diag.plot_residuals_direct(y_pred_train, weighted_residuals)
        diag.plot_qq(weighted_residuals)
        plt.show()
