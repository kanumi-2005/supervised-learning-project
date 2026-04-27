import numpy as np
from sklearn.base import ClassifierMixin, BaseEstimator


class KernelLogisticRegression(ClassifierMixin, BaseEstimator):
    """
    Kernel Logistic Regression classifier using the RBF kernel.

    This implementation performs multiclass classification using a
    kernelized version of logistic regression. The model operates in the
    dual space using the kernel trick, avoiding explicit feature mapping.

    The optimization is performed using gradient descent on the dual
    parameters (alpha) and bias (b).

    Parameters
    ----------
    gamma : float, default=1.0
        Parameter for the RBF kernel. Controls the width of the kernel.

    lr : float, default=0.1
        Learning rate used in gradient descent optimization.

    epochs : int, default=1000
        Number of iterations for training.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Unique class labels.

    X : ndarray of shape (n_samples, n_features)
        Training data stored for kernel computation.

    K : ndarray of shape (n_samples, n_samples)
        Kernel matrix computed on the training data.

    alpha : ndarray of shape (n_samples, n_classes)
        Dual coefficients learned during training.

    b : ndarray of shape (n_classes,)
        Bias terms for each class.

    is_fitted_ : bool
        Whether the model has been fitted.

    Notes
    -----
    This implementation uses a full kernel matrix, which may be memory
    intensive for large datasets (O(n^2) storage).

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[1, 2], [2, 3], [3, 4], [5, 6]])
    >>> y = np.array([0, 0, 1, 1])
    >>> clf = KernelLogisticRegression(gamma=0.5, epochs=200)
    >>> clf.fit(X, y)
    KernelLogisticRegression(...)
    >>> clf.predict(X)
    array([0, 0, 1, 1])
    """

    def __init__(self, gamma=1.0, lr=0.1, epochs=1000):
        """
        Initialize the Kernel Logistic Regression model.

        Parameters
        ----------
        gamma : float, default=1.0
            Parameter for the RBF kernel.

        lr : float, default=0.1
            Learning rate.

        epochs : int, default=1000
            Number of training iterations.
        """
        self.gamma = gamma
        self.lr = lr
        self.epochs = epochs

    def rbf_kernel(self, X1, X2):
        """
        Compute the RBF (Gaussian) kernel matrix.

        Parameters
        ----------
        X1 : ndarray of shape (n_samples_1, n_features)
            First input data.

        X2 : ndarray of shape (n_samples_2, n_features)
            Second input data.

        Returns
        -------
        K : ndarray of shape (n_samples_1, n_samples_2)
            Kernel matrix.
        """
        X1_sq = np.sum(X1**2, axis=1).reshape(-1, 1)
        X2_sq = np.sum(X2**2, axis=1)
        dist = X1_sq + X2_sq - 2 * X1 @ X2.T
        return np.exp(-self.gamma * dist)

    def softmax(self, Z):
        """
        Compute the softmax function.

        Parameters
        ----------
        Z : ndarray of shape (n_samples, n_classes)
            Input scores.

        Returns
        -------
        P : ndarray of shape (n_samples, n_classes)
            Probability distribution over classes.
        """
        Z = Z - np.max(Z, axis=1, keepdims=True)
        expZ = np.exp(Z)
        return expZ / np.sum(expZ, axis=1, keepdims=True)

    def fit(self, X, y):
        """
        Fit the Kernel Logistic Regression model.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.

        y : ndarray of shape (n_samples,)
            Target labels.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        n, d = X.shape
        self.classes_ = np.unique(y)
        c = len(self.classes_)

        y_idx = np.array(
            [np.where(self.classes_ == label)[0][0] for label in y])

        self.X = X
        self.K = self.rbf_kernel(X, X)

        self.alpha = np.zeros((n, c))
        self.b = np.zeros(c)

        for _ in range(self.epochs):
            # forward
            f = self.K @ self.alpha + self.b

            p = self.softmax(f)

            # one-hot labels
            y_onehot = np.zeros_like(p)
            y_onehot[np.arange(n), y_idx] = 1

            grad_f = (p - y_onehot)  # (n, c)

            # backprop
            grad_alpha = self.K.T @ grad_f
            grad_b = np.sum(grad_f, axis=0)

            self.alpha -= self.lr * grad_alpha
            self.b -= self.lr * grad_b

        self.is_fitted_ = True
        return self

    def predict(self, X):
        """
        Predict class labels for samples in X.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        K_test = self.rbf_kernel(X, self.X)

        f = K_test @ self.alpha + self.b
        p = self.softmax(f)

        return self.classes_[np.argmax(p, axis=1)]
