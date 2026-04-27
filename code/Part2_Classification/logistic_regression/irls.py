import numpy as np
import time
from sklearn.base import ClassifierMixin
from sklearn.metrics import log_loss
from ..base.basemodel import BaseModel


def sigmoid(x):
    x = np.clip(x, -50, 50)
    return 1 / (1 + np.exp(-x))


class IRLSClassifier(ClassifierMixin, BaseModel):
    """
    Iteratively Reweighted Least Squares (IRLS) binary classifier.

    IRLSClassifier fits a logistic regression model for binary
    classification using the Newton-Raphson optimization method.

    The model minimizes negative log-likelihood with L2 regularization.

    Parameters
    ----------
    max_iter : int, default=20
        Maximum number of IRLS iterations.

    reg_lambda : float, default=1e-4
        L2 regularization strength added to the Hessian diagonal.

    store_history : bool, default=True
        Whether to store training/validation loss and time history.

    Attributes
    ----------
    classes_ : ndarray of shape (2,)
        Unique class labels.

    w : ndarray of shape (n_features + 1,)
        Model weights including intercept.

    n_iter_ : int
        Number of iterations performed.

    train_loss_history_ : list
        Training loss per iteration (if enabled).

    val_loss_history_ : list
        Validation loss per iteration (if provided).

    time_history_ : list
        Elapsed time per iteration.

    Notes
    -----
    IRLS solves a sequence of weighted least squares problems
    using second-order optimization.

    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=100, n_features=5,
    ...                            random_state=0)
    >>> clf = IRLSClassifier(max_iter=3)
    >>> list(clf.fit(X, y))
    >>> clf.predict(X)[:5]
    """

    def __init__(self, max_iter=20, reg_lambda=1e-4, store_history=True):
        self.max_iter = max_iter
        self.reg_lambda = reg_lambda
        self.store_history = store_history

    def _encode_labels(self, y):
        """
        Encode binary labels into {0, 1} format.

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Target labels.

        Returns
        -------
        y_bin : ndarray of shape (n_samples,)
            Binary encoded labels.

        Raises
        ------
        ValueError
            If number of classes is not 2.
        """
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("IRLSClassifier only supports binary")
        return (y == self.classes_[1]).astype(float)

    def _init_params(self, X, y):
        """
        Initialize model parameters.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data.

        y : array-like
            Target labels.

        Returns
        -------
        None
        """
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features + 1)
        self.y_bin_ = self._encode_labels(y)

    def _loss(self, X, y):
        """
        Compute log loss.

        Parameters
        ----------
        X : array-like
            Input data.

        y : array-like
            True labels.

        Returns
        -------
        loss : float
            Log loss value.
        """
        probs = self.predict_proba(X)
        return log_loss(y, probs)

    def _fit(self, X, y, **kwargs):
        """
        Fit IRLS classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,)
            Binary target labels.

        X_val : array-like, optional
            Validation data.

        y_val : array-like, optional
            Validation labels.

        Yields
        ------
        dict
            Dictionary with iteration, train loss, val loss,
            and elapsed time.

        Returns
        -------
        self : object
            Fitted model after training.
        """
        X_val = kwargs.get("X_val", None)
        y_val = kwargs.get("y_val", None)

        self._init_params(X, y)

        n_samples = X.shape[0]
        X_design = np.c_[np.ones(n_samples), X]

        if self.store_history:
            self.train_loss_history_ = []
            self.val_loss_history_ = []
            self.time_history_ = []

        start_time = time.perf_counter()

        for it in range(self.max_iter):
            z = X_design @ self.w
            p = sigmoid(z)

            W = np.clip(p * (1 - p), 1e-8, None)
            H = X_design.T @ (W[:, None] * X_design) + \
                self.reg_lambda * np.eye(X_design.shape[1])
            g = X_design.T @ (p - self.y_bin_)

            try:
                delta = np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                delta = np.linalg.pinv(H) @ g

            self.w -= delta

            train_loss = self._loss(X, y)

            val_loss = None
            if X_val is not None and y_val is not None:
                val_loss = self._loss(X_val, y_val)

            elapsed = time.perf_counter() - start_time

            if self.store_history:
                self.train_loss_history_.append(train_loss)
                if val_loss is not None:
                    self.val_loss_history_.append(val_loss)
                self.time_history_.append(elapsed)

            yield {
                "iter": it,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "time": elapsed
            }

        self.n_iter_ = self.max_iter

    def _predict(self, X):
        """
        Predict class labels.

        Parameters
        ----------
        X : array-like
            Input data.

        Returns
        -------
        y_pred : ndarray
            Predicted class labels.
        """
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]

    def predict_proba(self, X):
        """
        Predict class probabilities.

        Parameters
        ----------
        X : array-like
            Input data.

        Returns
        -------
        proba : ndarray of shape (n_samples, 2)
            Class probabilities.
        """
        n_samples = X.shape[0]
        X_design = np.c_[np.ones(n_samples), X]
        p = sigmoid(X_design @ self.w)
        return np.c_[1 - p, p]

    @property
    def intercept_(self):
        """float : Intercept term."""
        return self.w[0]

    @property
    def coef_(self):
        """ndarray : Model coefficients (excluding intercept)."""
        return self.w[1:]
