from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from ..base.basemodel import BaseModel


class QDA(BaseModel):
    """
    Quadratic Discriminant Analysis (QDA) classifier.

    This class is a wrapper around
    `sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis`.
    It fits class-specific Gaussian distributions with distinct
    covariance matrices and applies Bayes' rule for classification.

    Parameters
    ----------
    reg_param : float, default=0.001
        Regularization parameter added to the diagonal of covariance
        matrices. Helps improve numerical stability when covariance
        matrices are ill-conditioned.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Unique class labels.

    means_ : ndarray of shape (n_classes, n_features)
        Class-wise feature means.

    priors_ : ndarray of shape (n_classes,)
        Class prior probabilities.

    covariances_ : list of ndarray
        Covariance matrices for each class.

    model : QuadraticDiscriminantAnalysis
        Internal sklearn QDA model.

    Notes
    -----
    QDA assumes that each class follows a Gaussian distribution with
    its own covariance matrix. Unlike LDA, QDA allows more flexible
    decision boundaries but may require more data to estimate reliably.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(
    ...     n_samples=100, n_features=4, n_classes=2, random_state=0
    ... )
    >>> clf = QDA().fit(X, y)
    >>> clf.score(X, y)
    1.0
    >>> clf.predict(X[:5])
    array([0, 1, 0, 1, 0])
    >>> clf.predict_proba(X[:2])
    array([[0.8, 0.2],
           [0.3, 0.7]])
    """

    def __init__(self, reg_param=0.001):
        """
        Initialize the QDA estimator.

        Parameters
        ----------
        reg_param : float, default=0.001
            Regularization parameter for covariance matrices.
        """
        self.reg_param = reg_param
        self.model = QuadraticDiscriminantAnalysis(
            reg_param=reg_param,
            store_covariance=True
        )

    def _fit(self, X, y, **kwargs):
        """
        Fit the QDA model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,)
            Target labels.

        **kwargs : dict
            Additional keyword arguments (ignored).

        Returns
        -------
        None
        """
        self.model.fit(X, y)
        self.classes_ = self.model.classes_
        self.means_ = self.model.means_
        self.priors_ = self.model.priors_
        self.covariances_ = self.model.covariance_

    def _predict(self, X):
        """
        Perform classification on samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        return self.model.predict(X)

    def predict_proba(self, X):
        """
        Estimate class probabilities for samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Posterior probabilities of each class.
        """
        return self.model.predict_proba(X)

    def score(self, X, y):
        """
        Return the mean accuracy on the given test data and labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        y : array-like of shape (n_samples,)
            True labels.

        Returns
        -------
        score : float
            Mean accuracy of predictions.
        """
        return self.model.score(X, y)
