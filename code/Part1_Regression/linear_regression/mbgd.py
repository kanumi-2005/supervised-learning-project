import numpy as np
from sklearn.base import RegressorMixin
from sklearn.metrics import mean_squared_error
from ..base.basegdmodel import BaseGDModel


class MBGD(RegressorMixin, BaseGDModel):
    """
    Mini-Batch Gradient Descent Regressor.

    MBGD fits a linear regression model using mini-batch gradient
    descent optimization. It supports multiple learning rate schedules
    and optional tracking of training/validation loss history.

    The model learns parameters w = (w0, w1, ..., wp), where w0 is the
    intercept term.

    Parameters
    ----------
    batch_size : int, default=64
        Number of samples per mini-batch.

    lr : float, default=0.01
        Initial learning rate.

    max_iter : int, default=150
        Number of training iterations (epochs).

    lr_sched : {"step_decay", "cosine_annealing", None}, default=None
        Learning rate schedule strategy.

    random_state : int, default=42
        Seed for reproducibility.

    min_lr : float, default=1e-4
        Minimum learning rate for cosine annealing.

    step_size : int, default=30
        Step size for step decay schedule.

    decay_factor : float, default=0.5
        Multiplicative decay factor for step decay.

    store_history : bool, default=True
        Whether to store loss history during training.

    Attributes
    ----------
    w : ndarray of shape (n_features + 1,)
        Model parameters including intercept.

    coef_ : ndarray of shape (n_features,)
        Estimated coefficients for features.

    intercept_ : float
        Intercept term.

    n_iter_ : int
        Number of training iterations performed.

    train_loss_history_ : list
        Training loss per iteration (if enabled).

    val_loss_history_ : list
        Validation loss per iteration (if available).

    Examples
    --------
    >>> import numpy as np
    >>> X = np.random.randn(100, 3)
    >>> y = X @ np.array([1, 2, 3]) + 5
    >>> model = MBGD(max_iter=5)
    >>> model.fit(X, y)
    >>> model.predict(X[:3])
    """

    def __init__(
        self,
        batch_size=64,
        lr=0.01,
        max_iter=150,
        lr_sched=None,
        random_state=42,
        min_lr=0.0001,
        step_size=30,
        decay_factor=0.5,
        store_history=True
    ):
        """
        Initialize MBGD model.

        Parameters
        ----------
        batch_size : int, default=64
            Mini-batch size.

        lr : float, default=0.01
            Learning rate.

        max_iter : int, default=150
            Number of iterations.

        lr_sched : str or None, default=None
            Learning rate schedule.

        random_state : int, default=42
            Random seed.

        min_lr : float, default=1e-4
            Minimum learning rate.

        step_size : int, default=30
            Step size for decay.

        decay_factor : float, default=0.5
            Decay factor.

        store_history : bool, default=True
            Store loss history.
        """
        super().__init__(
            lr=lr,
            max_iter=max_iter,
            store_history=store_history,
            batch_size=batch_size,
            random_state=random_state
        )
        self.lr_sched = lr_sched
        self.min_lr = min_lr
        self.step_size = step_size
        self.decay_factor = decay_factor

    def _init_params(self, X, y):
        """
        Initialize model parameters.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like

        Returns
        -------
        None
        """
        n_features = X.shape[1]
        self.w = np.zeros(n_features + 1)

    def _loss(self, X, y):
        """
        Compute mean squared error loss.

        Parameters
        ----------
        X : array-like
        y : array-like

        Returns
        -------
        float
            Loss value.
        """
        y_pred = self.w[0] + X @ self.w[1:]
        return mean_squared_error(y, y_pred)

    def _grad(self, X, y):
        """
        Compute gradient of loss.

        Parameters
        ----------
        X : array-like
        y : array-like

        Returns
        -------
        ndarray
            Gradient vector.
        """
        n_samples = X.shape[0]
        X_design = np.c_[np.ones(n_samples), X]
        return - X_design.T @ (y - X_design @ self.w) / n_samples

    def _update_params(self, grad, iteration):
        """
        Update model parameters.

        Parameters
        ----------
        grad : ndarray
            Gradient.

        iteration : int
            Current iteration.

        Returns
        -------
        None
        """
        self.w -= self._lr(iteration) * grad

    def _lr(self, iteration):
        """
        Compute learning rate at a given iteration.

        Parameters
        ----------
        iteration : int

        Returns
        -------
        float
        """
        if self.lr_sched is None:
            return self.lr
        elif self.lr_sched == "step_decay":
            return self.lr * (self.decay_factor **
                              (iteration // self.step_size))
        elif self.lr_sched == "cosine_annealing":
            return self.min_lr + 0.5 * (self.lr - self.min_lr) * \
                   (1 + np.cos(np.pi * iteration / self.max_iter))
        else:
            raise ValueError(f"Unknown lr_sched: {self.lr_sched}")

    def _extra_logs(self, X, y, grad, iter):
        """
        Extra logging information.

        Parameters
        ----------
        X : array-like
        y : array-like
        grad : ndarray or None
        iter : int

        Returns
        -------
        dict
        """
        return {"lr": self._lr(iter)}

    def _predict(self, X):
        """
        Predict target values.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        ndarray
            Predicted values.
        """
        return self.w[0] + X @ self.w[1:]

    @property
    def intercept_(self):
        """
        Intercept term.

        Returns
        -------
        float
        """
        return self.w[0]

    @property
    def coef_(self):
        """
        Model coefficients.

        Returns
        -------
        ndarray
        """
        return self.w[1:]


if __name__ == "__main__":
    from ..dataset import CaliforniaHousingDataset as Dataset
    from ..pipeline import get_pipeline
    from sklearn.metrics import mean_squared_error

    d = Dataset()
    d.split()

    model = get_pipeline(MBGD(lr_sched="step_decay"))
    model.fit(d.X_train, d.y_train)
    y_pred = model.predict(d.X_test)
    y_true = d.y_test

    mse = mean_squared_error(y_true, y_pred)

    print(f"MSE = {mse:.4f}")
