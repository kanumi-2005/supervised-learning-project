import numpy as np
from scipy.stats import norm
from sklearn.base import ClassifierMixin
from sklearn.metrics import log_loss
from ..base.basegdmodel import BaseGDModel


def normal_pdf(x):
    return norm.pdf(x)


def normal_cdf(x):
    return norm.cdf(x)


class ProbitClassifier(ClassifierMixin, BaseGDModel):
    """
    Probit Regression Classifier.

    This classifier models binary outcomes using a latent variable
    formulation with the probit link function (Gaussian CDF).

    The model assumes:
        P(y = 1 | x) = Phi(w^T x)
    where Phi is the standard normal cumulative distribution function.

    Optimization is performed using gradient descent variants
    inherited from BaseGDModel.

    Parameters
    ----------
    lr : float, default=0.01
        Learning rate for gradient updates.

    max_iter : int, default=50
        Maximum number of training iterations.

    batch_size : int, default=1024
        Mini-batch size for stochastic gradient descent.

    lr_sched : {"step_decay", None}, default="step_decay"
        Learning rate scheduling strategy.

    step_size : int, default=10
        Number of iterations per learning rate decay step.

    decay_factor : float, default=0.5
        Multiplicative factor for learning rate decay.

    random_state : int, default=42
        Seed for reproducible shuffling.

    store_history : bool, default=True
        If True, stores training history.

    Attributes
    ----------
    w : ndarray of shape (n_features + 1,)
        Model parameters including intercept.

    classes_ : ndarray of shape (2,)
        Unique class labels.

    y_bin_ : ndarray of shape (n_samples,)
        Binary encoded labels.

    intercept_ : float
        Intercept term.

    coef_ : ndarray of shape (n_features,)
        Model coefficients.

    n_iter_ : int
        Number of iterations run during training.

    Notes
    -----
    The probit model uses a Gaussian latent variable assumption:
    z = w^T x, and prediction is based on Phi(z).

    Optimization is performed via gradient descent.

    Examples
    --------
    >>> model = ProbitClassifier(lr=0.01, max_iter=100)
    >>> model.fit(X_train, y_train)
    >>> preds = model.predict(X_test)
    >>> proba = model.predict_proba(X_test)
    """

    def __init__(
        self,
        lr=0.01,
        max_iter=50,
        batch_size=1024,
        lr_sched="step_decay",
        step_size=10,
        decay_factor=0.5,
        random_state=42,
        store_history=True
    ):
        """
        Initialize Probit Classifier.

        Parameters
        ----------
        lr : float, default=0.01
            Learning rate.

        max_iter : int, default=50
            Number of training iterations.

        batch_size : int, default=1024
            Mini-batch size.

        lr_sched : {"step_decay", None}, default="step_decay"
            Learning rate schedule type.

        step_size : int, default=10
            Iterations per decay step.

        decay_factor : float, default=0.5
            Learning rate decay factor.

        random_state : int, default=42
            Random seed.

        store_history : bool, default=True
            Whether to store training history.
        """
        super().__init__(
            lr=lr,
            max_iter=max_iter,
            store_history=store_history,
            batch_size=batch_size,
            random_state=random_state
        )
        self.lr_sched = lr_sched
        self.step_size = step_size
        self.decay_factor = decay_factor

    def _encode_labels(self, y):
        """
        Encode binary labels to {0, 1} format.

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Class labels.

        Returns
        -------
        ndarray of shape (n_samples,)
            Binary encoded labels.

        Raises
        ------
        ValueError
            If number of classes is not equal to 2.
        """
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("Probit only supports binary classification")
        return (y == self.classes_[1]).astype(float)

    def _init_params(self, X, y):
        """
        Initialize model parameters.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,)
            Target labels.

        Returns
        -------
        None
        """
        n_features = X.shape[1]
        self.w = np.zeros(n_features + 1)
        self.y_bin_ = self._encode_labels(y)

    def _loss(self, X, y):
        """
        Compute log-loss.

        Parameters
        ----------
        X : array-like
            Input data.

        y : array-like
            True labels.

        Returns
        -------
        float
            Log-loss value.
        """
        probs = self.predict_proba(X)
        return log_loss(y, probs)

    def _grad(self, X, y):
        """
        Compute gradient of probit log-likelihood.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input batch.

        y : array-like of shape (n_samples,)
            Target batch labels.

        Returns
        -------
        ndarray
            Gradient vector.
        """
        n_samples = X.shape[0]
        X_design = np.c_[np.ones(n_samples), X]

        y_bin = (y == self.classes_[1]).astype(float)

        z = X_design @ self.w
        p = normal_cdf(z)
        pdf = normal_pdf(z)

        eps = 1e-10
        p = np.clip(p, eps, 1 - eps)

        diff = (p - y_bin) * (pdf / (p * (1 - p)))
        grad = X_design.T @ diff
        return grad / n_samples

    def _update_params(self, grad, iteration):
        """
        Update model parameters using gradient descent.

        Parameters
        ----------
        grad : ndarray
            Gradient of loss.

        iteration : int
            Current iteration index.

        Returns
        -------
        None
        """
        self.w -= self._lr(iteration) * grad

    def _lr(self, iteration):
        """
        Compute learning rate at given iteration.

        Parameters
        ----------
        iteration : int
            Current iteration index.

        Returns
        -------
        float
            Learning rate value.
        """
        if self.lr_sched is None:
            return self.lr
        elif self.lr_sched == "step_decay":
            return self.lr * (
                self.decay_factor ** (iteration // self.step_size)
            )
        else:
            raise ValueError(f"Unknown lr_sched: {self.lr_sched}")

    def _extra_logs(self, X, y, grad, iter):
        """
        Additional logging information.

        Parameters
        ----------
        X : array-like
            Input data.

        y : array-like
            Target labels.

        grad : array-like or None
            Gradient value.

        iter : int
            Current iteration index.

        Returns
        -------
        dict
            Dictionary with learning rate.
        """
        return {"lr": self._lr(iter)}

    def _predict(self, X):
        """
        Predict class labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        ndarray
            Predicted class labels.
        """
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]

    def predict_proba(self, X):
        """
        Predict class probabilities.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        ndarray of shape (n_samples, 2)
            Class probabilities.
        """
        n_samples = X.shape[0]
        X_design = np.c_[np.ones(n_samples), X]

        z = X_design @ self.w
        p = normal_cdf(z)

        return np.c_[1 - p, p]

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
