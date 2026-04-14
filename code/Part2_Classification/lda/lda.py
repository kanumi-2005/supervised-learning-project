import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


class LDA(ClassifierMixin, BaseEstimator):
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.model = LinearDiscriminantAnalysis(
            store_covariance=True,
            n_components=n_components
        )

    def fit(self, X, y):
        self.model.fit(X, y)
        self.classes_ = self.model.classes_
        self.means_ = self.model.means_
        self.priors_ = self.model.priors_
        self.covariance_ = self.model.covariance_
        return self

    def fit_transform(self, X):
        return self.model.fit_transform(X)

    def transform(self, X):
        return self.model.transform(X)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def score(self, X, y):
        return self.model.score(X, y)

    def plot2D(self, X, y, padding=1):
        if self.n_components != 2:
            raise ValueError("plot2D requires n_components = 2")

        X_lda = self.transform(X)

        clf = LinearDiscriminantAnalysis(n_components=2)
        clf.fit(X_lda, y)

        x_min, x_max = X_lda[:, 0].min() - padding, X_lda[:, 0].max() + padding
        y_min, y_max = X_lda[:, 1].min() - padding, X_lda[:, 1].max() + padding

        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 300),
            np.linspace(y_min, y_max, 300)
        )

        grid = np.c_[xx.ravel(), yy.ravel()]
        Z = clf.predict(grid)
        Z = Z.reshape(xx.shape)

        fig, ax = plt.subplots(layout="constrained")

        ax.contour(xx, yy, Z, colors='k', linewidths=1)

        for label in np.unique(y):
            plt.scatter(
                X_lda[y == label, 0],
                X_lda[y == label, 1],
                label=f"Class {label}",
                alpha=0.5
            )

        ax.set_xlabel("LD1")
        ax.set_ylabel("LD2")
        ax.set_title("LDA Decision Boundary")
        ax.legend(loc='best')
        plt.show()
