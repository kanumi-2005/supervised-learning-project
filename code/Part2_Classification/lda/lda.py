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

    def plot2D(self, X, y):
        if self.n_components != 2:
            raise ValueError("plot2D is only supported when n_components = 2")

        X_lda = self.transform(X)

        plt.figure()
        for label in np.unique(y):
            plt.scatter(
                X_lda[y == label, 0],
                X_lda[y == label, 1],
                label=f"Class {label}"
            )

        x_min, x_max = X_lda[:, 0].min() - 1, X_lda[:, 0].max() + 1
        y_min, y_max = X_lda[:, 1].min() - 1, X_lda[:, 1].max() + 1

        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 200),
            np.linspace(y_min, y_max, 200)
        )

        Z = self.model.predict(
            self.model.inverse_transform(np.c_[xx.ravel(), yy.ravel()])
        )
        Z = Z.reshape(xx.shape)

        plt.contour(xx, yy, Z)

        plt.xlabel("LD1")
        plt.ylabel("LD2")
        plt.legend()
        plt.title("LDA Decision Boundary (2D)")
        plt.show()
