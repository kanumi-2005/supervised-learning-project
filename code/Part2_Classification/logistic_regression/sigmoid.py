import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.metrics import log_loss
from ..base.basegdmodel import BaseGDModel


def sigmoid(x):
    """
    Compute the sigmoid function with numerical stability.

    Parameters
    ----------
    x : array-like
        Input values.

    Returns
    -------
    array-like
        Sigmoid applied element-wise.
    """
    x = np.clip(x, -50, 50)
    return 1 / (1 + np.exp(-x))


class SigmoidClassifier(ClassifierMixin, BaseGDModel):
    """
    Logistic Regression classifier trained with gradient descent.

    This classifier implements binary logistic regression using a
    sigmoid activation function and gradient-based optimization.

    It supports learning rate scheduling, mini-batch training,
    and optional history tracking for loss and training dynamics.

    Parameters
    ----------
    lr : float, default=0.01
        Learning rate for gradient descent updates.

    max_iter : int, default=50
        Number of training iterations (epochs).

    batch_size : int, default=1024
        Number of samples per mini-batch.

    lr_sched : str or None, default="step_decay"
        Learning rate schedule type.

    step_size : int, default=10
        Number of iterations per decay step.

    decay_factor : float, default=0.5
        Multiplicative factor for step decay scheduling.

    random_state : int, default=42
        Random seed for reproducibility.

    store_history : bool, default=True
        Whether to store training and validation history.

    Attributes
    ----------
    w : ndarray of shape (n_features + 1,)
        Model weights including bias term.

    classes_ : ndarray of shape (2,)
        Unique class labels.

    coef_ : ndarray of shape (n_features,)
        Feature coefficients excluding intercept.

    intercept_ : float
        Bias term of the model.

    train_loss_history_ : list
        Training loss per iteration.

    val_loss_history_ : list
        Validation loss per iteration.

    time_history_ : list
        Time elapsed per iteration.

    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=1000, n_features=5)
    >>> clf = SigmoidClassifier(max_iter=10)
    >>> clf.fit(X, y)
    >>> preds = clf.predict(X)
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
        Initialize the SigmoidClassifier.

        Parameters
        ----------
        lr : float
            Learning rate.

        max_iter : int
            Number of training iterations.

        batch_size : int
            Mini-batch size.

        lr_sched : str or None
            Learning rate schedule.

        step_size : int
            Steps per decay.

        decay_factor : float
            Decay multiplier.

        random_state : int
            Random seed.

        store_history : bool
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
        Encode labels into binary format.

        Parameters
        ----------
        y : array-like
            Target labels.

        Returns
        -------
        ndarray
            Binary encoded labels.
        """
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("SigmoidClassifier only supports binary")
        return (y == self.classes_[1]).astype(float)

    def _init_params(self, X, y):
        """
        Initialize model parameters.

        Parameters
        ----------
        X : ndarray
            Input features.

        y : array-like
            Target labels.
        """
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features + 1)
        self.y_bin_ = self._encode_labels(y)

    def _loss(self, X, y):
        """
        Compute log loss.

        Parameters
        ----------
        X : ndarray
            Input features.

        y : array-like
            True labels.

        Returns
        -------
        float
            Log loss value.
        """
        probs = self.predict_proba(X)
        return log_loss(y, probs)

    def _grad(self, X, y):
        """
        Compute gradient of loss function.

        Parameters
        ----------
        X : ndarray
            Input features.

        y : array-like
            True labels.

        Returns
        -------
        ndarray
            Gradient vector.
        """
        n_samples = X.shape[0]
        X_design = np.c_[np.ones(n_samples), X]

        y_bin = (y == self.classes_[1]).astype(float)
        p = sigmoid(X_design @ self.w)

        return X_design.T @ (p - y_bin) / n_samples

    def _update_params(self, grad, iteration):
        """
        Update model parameters using gradient.

        Parameters
        ----------
        grad : ndarray
            Gradient vector.

        iteration : int
            Current iteration index.
        """
        self.w -= self._lr(iteration) * grad

    def _lr(self, iteration):
        """
        Compute learning rate for current iteration.

        Parameters
        ----------
        iteration : int
            Iteration index.

        Returns
        -------
        float
            Learning rate.
        """
        if self.lr_sched is None:
            return self.lr
        elif self.lr_sched == "step_decay":
            return self.lr * (self.decay_factor ** (
                iteration // self.step_size
            ))
        else:
            raise ValueError(f"Unknown lr_sched: {self.lr_sched}")

    def _extra_logs(self, X, y, grad, iter):
        """
        Additional logs during training.

        Returns
        -------
        dict
            Dictionary containing learning rate.
        """
        return {"lr": self._lr(iter)}

    def _predict(self, X):
        """
        Predict class labels.

        Parameters
        ----------
        X : ndarray
            Input features.

        Returns
        -------
        ndarray
            Predicted class labels.
        """
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]

    def predict_proba(self, X):
        """
        Compute class probabilities.

        Parameters
        ----------
        X : ndarray
            Input features.

        Returns
        -------
        ndarray
            Probability matrix of shape (n_samples, 2).
        """
        n_samples = X.shape[0]
        X_design = np.c_[np.ones(n_samples), X]
        p = sigmoid(X_design @ self.w)
        return np.c_[1 - p, p]

    @property
    def intercept_(self):
        """Intercept (bias term) of the model."""
        return self.w[0]

    @property
    def coef_(self):
        """Coefficient vector excluding intercept."""
        return self.w[1:]
