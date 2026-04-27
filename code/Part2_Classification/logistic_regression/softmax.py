import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.metrics import log_loss
from ..base.basegdmodel import BaseGDModel


def softmax(z):
    """
    Compute softmax probabilities in a numerically stable way.

    Parameters
    ----------
    z : ndarray of shape (n_samples, n_classes)
        Input logits.

    Returns
    -------
    ndarray of shape (n_samples, n_classes)
        Normalized probability distribution.
    """
    z = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


class SoftmaxClassifier(ClassifierMixin, BaseGDModel):
    """
    Multiclass logistic regression classifier using softmax.

    This classifier implements multinomial logistic regression trained
    with mini-batch gradient descent. It uses the softmax function to
    model class probabilities and cross-entropy loss for optimization.

    Parameters
    ----------
    lr : float, default=0.01
        Learning rate for gradient updates.

    max_iter : int, default=50
        Number of training iterations.

    batch_size : int, default=1024
        Size of each mini-batch.

    lr_sched : str or None, default="step_decay"
        Learning rate schedule strategy.

    step_size : int, default=10
        Number of iterations per decay step.

    decay_factor : float, default=0.5
        Multiplicative decay factor for learning rate.

    random_state : int, default=42
        Random seed for reproducibility.

    store_history : bool, default=True
        Whether to store training history.

    Attributes
    ----------
    W : ndarray of shape (n_features + 1, n_classes)
        Model weight matrix including bias row.

    classes_ : ndarray of shape (n_classes,)
        Unique class labels.

    class_to_index_ : dict
        Mapping from class label to integer index.

    n_classes_ : int
        Number of target classes.

    coef_ : ndarray of shape (n_features, n_classes)
        Feature coefficients excluding intercept.

    intercept_ : ndarray of shape (n_classes,)
        Bias term for each class.

    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=200, n_classes=3)
    >>> clf = SoftmaxClassifier(max_iter=10)
    >>> clf.fit(X, y)
    >>> clf.predict(X[:5])
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
        Initialize SoftmaxClassifier.

        Parameters
        ----------
        lr : float
            Learning rate.

        max_iter : int
            Number of iterations.

        batch_size : int
            Mini-batch size.

        lr_sched : str or None
            Learning rate schedule.

        step_size : int
            Step interval for decay.

        decay_factor : float
            Learning rate decay factor.

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
        Encode labels into integer indices.

        Parameters
        ----------
        y : array-like
            Target labels.

        Returns
        -------
        ndarray
            Encoded label indices.
        """
        self.classes_ = np.unique(y)
        self.class_to_index_ = {c: i for i, c in enumerate(self.classes_)}
        return np.array([self.class_to_index_[c] for c in y])

    def _one_hot(self, y, n_classes):
        """
        Convert labels to one-hot encoding.

        Parameters
        ----------
        y : ndarray
            Label indices.

        n_classes : int
            Number of classes.

        Returns
        -------
        ndarray
            One-hot encoded matrix.
        """
        one_hot = np.zeros((len(y), n_classes))
        one_hot[np.arange(len(y)), y] = 1
        return one_hot

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
        y_encoded = self._encode_labels(y)
        self.n_classes_ = len(self.classes_)

        self.X_ = X
        self.y_encoded_ = y_encoded

        self.W = np.zeros((n_features + 1, self.n_classes_))

    def _loss(self, X, y):
        """
        Compute cross-entropy loss.

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
        Compute gradient of softmax loss.

        Parameters
        ----------
        X : ndarray
            Input features.

        y : array-like
            True labels.

        Returns
        -------
        ndarray
            Gradient matrix.
        """
        n_samples = X.shape[0]
        X_design = np.c_[np.ones(n_samples), X]

        y_encoded = np.array(
            [self.class_to_index_[c] for c in y]
        )

        probs = softmax(X_design @ self.W)
        probs[np.arange(n_samples), y_encoded] -= 1

        return (X_design.T @ probs) / n_samples

    def _update_params(self, grad, iteration):
        """
        Update parameters using gradient descent.

        Parameters
        ----------
        grad : ndarray
            Gradient matrix.

        iteration : int
            Current iteration index.
        """
        self.W -= self._lr(iteration) * grad

    def _lr(self, iteration):
        """
        Compute learning rate for iteration.

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
            return self.lr * (
                self.decay_factor ** (iteration // self.step_size)
            )
        else:
            raise ValueError(f"Unknown lr_sched: {self.lr_sched}")

    def _extra_logs(self, X, y, grad, iter):
        """
        Additional training logs.

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
        indices = np.argmax(probs, axis=1)
        return self.classes_[indices]

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
            Probability matrix of shape (n_samples, n_classes).
        """
        n_samples = X.shape[0]
        X_design = np.c_[np.ones(n_samples), X]
        return softmax(X_design @ self.W)

    @property
    def intercept_(self):
        """Intercept vector for each class."""
        return self.W[0]

    @property
    def coef_(self):
        """Coefficient matrix excluding intercept row."""
        return self.W[1:]
