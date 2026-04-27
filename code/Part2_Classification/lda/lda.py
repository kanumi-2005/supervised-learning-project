import numpy as np
import matplotlib.pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from ..base.basemodel import BaseModel


class LDA(BaseModel):
    """
    Linear Discriminant Analysis (LDA) wrapper.

    This class provides a thin wrapper around
    sklearn.discriminant_analysis.LinearDiscriminantAnalysis
    with additional convenience methods and plotting utilities.

    LDA is a supervised dimensionality reduction technique that
    projects data onto a lower-dimensional space while maximizing
    class separability.

    Parameters
    ----------
    n_components : int, default=None
        Number of components to keep. If None, will be set to
        min(n_classes - 1, n_features).

    Attributes
    ----------
    model : LinearDiscriminantAnalysis
        Underlying sklearn LDA model.

    classes_ : ndarray of shape (n_classes,)
        Class labels known to the classifier.

    means_ : ndarray of shape (n_classes, n_features)
        Class-wise means.

    priors_ : ndarray of shape (n_classes,)
        Class prior probabilities.

    covariance_ : ndarray of shape (n_features, n_features)
        Weighted within-class covariance matrix.

    Notes
    -----
    This implementation relies on sklearn's LDA and exposes
    additional helper methods for transformation and visualization.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.datasets import load_iris
    >>> X, y = load_iris(return_X_y=True)
    >>> lda = LDA(n_components=2)
    >>> lda._fit(X, y)
    >>> X_trans = lda.transform(X)
    >>> lda.score(X, y)
    1.0
    """

    def __init__(self, n_components=None):
        """
        Initialize the LDA model.

        Parameters
        ----------
        n_components : int, default=None
            Number of components to keep.
        """
        self.n_components = n_components
        self.model = LinearDiscriminantAnalysis(
            store_covariance=True,
            n_components=n_components
        )

    def _fit(self, X, y, **kwargs):
        """
        Fit the LDA model.

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
        self.covariance_ = self.model.covariance_

    def _predict(self, X):
        """
        Predict class labels for samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        return self.model.predict(X)

    def fit_transform(self, X, y=None):
        """
        Fit the model and apply dimensionality reduction.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,), default=None
            Target labels.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
            Transformed data.
        """
        return self.model.fit_transform(X, y)

    def transform(self, X):
        """
        Project data to the LDA subspace.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
            Transformed data.
        """
        return self.model.transform(X)

    def predict_proba(self, X):
        """
        Estimate class probabilities for samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Predicted class probabilities.
        """
        return self.model.predict_proba(X)

    def score(self, X, y):
        """
        Return the mean accuracy on the given test data.

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

    def plot2D(self, X, y):
        """
        Plot decision boundary in 2D LDA space.

        This method projects the data into 2D using LDA and
        visualizes the decision boundary along with class points.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        y : array-like of shape (n_samples,)
            Target labels.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If n_components is not equal to 2.
        """
        if self.n_components != 2:
            raise ValueError("plot2D requires n_components = 2")

        X_lda = self.transform(X)

        clf = LinearDiscriminantAnalysis(n_components=2)
        clf.fit(X_lda, y)

        fig, ax = plt.subplots(layout="constrained")

        DecisionBoundaryDisplay.from_estimator(
            clf,
            X_lda,
            response_method="predict",
            grid_resolution=300,
            ax=ax,
            alpha=0.3
        )

        for label in np.unique(y):
            ax.scatter(
                X_lda[y == label, 0],
                X_lda[y == label, 1],
                label=f"Class {label}",
                alpha=0.5
            )

        ax.set_title("LDA Decision Boundary")
        ax.legend(loc="best")
        plt.show()
