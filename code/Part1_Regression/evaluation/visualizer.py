import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class Visualizer:
    """
    Utility class for visualizing machine learning model performance.

    This class provides methods for plotting learning curves,
    residual plots, and predicted vs actual comparisons for
    regression models.

    Examples
    --------
    >>> vis = Visualizer()
    >>> vis.plot_learning_curve(model, X, y)
    >>> vis.plot_residuals(y_true, y_pred)
    >>> vis.plot_pred_vs_actual(y_true, y_pred)
    """

    def plot_learning_curve(self, model, X, y, ax=None, cv=5):
        """
        Plot learning curve of a model.

        The learning curve shows training and validation loss
        as a function of training set size.

        Parameters
        ----------
        model : estimator
            Machine learning model implementing fit/predict.

        X : array-like
            Input features.

        y : array-like
            Target values.

        ax : matplotlib.axes.Axes, default=None
            Axis to plot on. If None, a new figure is created.

        cv : int, default=5
            Number of cross-validation folds.

        Returns
        -------
        ax : matplotlib.axes.Axes
            Axis with plotted learning curve.
        """
        train_sizes, train_scores, val_scores = learning_curve(
            model,
            X,
            y,
            cv=cv,
            scoring="neg_mean_squared_error",
            train_sizes=np.linspace(0.1, 1.0, 10),
            shuffle=True,
            random_state=42
        )

        train_loss = -np.mean(train_scores, axis=1)
        val_loss = -np.mean(val_scores, axis=1)

        if ax is None:
            _, ax = plt.subplots(layout="constrained")

        ax.plot(train_sizes, train_loss, label="Train Loss")
        ax.plot(train_sizes, val_loss, label="Validation Loss")

        ax.set_xlabel("Training Size")
        ax.set_ylabel("MSE")
        ax.set_title("Learning Curve")
        ax.legend()
        ax.set_axisbelow(True)
        ax.grid()

        return ax

    def plot_residuals(self, y_true, y_pred, ax=None):
        """
        Plot residual errors of predictions.

        Residuals are defined as (y_true - y_pred). This plot
        helps diagnose model bias and heteroscedasticity.

        Parameters
        ----------
        y_true : array-like
            Ground truth values.

        y_pred : array-like
            Predicted values.

        ax : matplotlib.axes.Axes, default=None
            Axis to plot on. If None, a new figure is created.

        Returns
        -------
        ax : matplotlib.axes.Axes
            Axis with residual plot.
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

    def plot_pred_vs_actual(self, y_true, y_pred, ax=None):
        """
        Plot predicted values against actual values.

        This plot visualizes how close predictions are to the
        ideal diagonal line y = x.

        Parameters
        ----------
        y_true : array-like
            Ground truth values.

        y_pred : array-like
            Predicted values.

        ax : matplotlib.axes.Axes, default=None
            Axis to plot on. If None, a new figure is created.

        Returns
        -------
        ax : matplotlib.axes.Axes
            Axis with scatter plot.
        """
        if ax is None:
            _, ax = plt.subplots(layout="constrained")

        ax.scatter(y_true, y_pred)

        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())

        ax.plot([min_val, max_val], [min_val, max_val],
                linestyle="--")

        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title("Predicted vs Actual")
        ax.set_axisbelow(True)
        ax.grid()

        return ax


if __name__ == "__main__":
    from ..dataset import CaliforniaHousingDataset as Dataset
    from ..pipeline import get_pipeline
    from ..linear_regression.ols import OLS
    from sklearn.compose import TransformedTargetRegressor
    from ..regularization.lasso_regression import LassoRegression

    d = Dataset()
    d.split()

    model = get_pipeline(OLS())

    visualizer = Visualizer()

    visualizer.plot_learning_curve(model, d.X_train, d.y_train)
    model.fit(d.X_train, d.y_train)
    y_pred = model.predict(d.X_test)
    y_true = d.y_test
    visualizer.plot_residuals(y_true, y_pred)
    visualizer.plot_pred_vs_actual(y_true, y_pred)
    plt.show()
