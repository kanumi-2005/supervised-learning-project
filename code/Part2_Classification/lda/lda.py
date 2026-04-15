import numpy as np
import matplotlib.pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from ..base.basemodel import BaseModel


class LDA(BaseModel):
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.model = LinearDiscriminantAnalysis(
            store_covariance=True,
            n_components=n_components
        )

    def _fit(self, X, y, **kwargs):
        self.model.fit(X, y)
        self.classes_ = self.model.classes_
        self.means_ = self.model.means_
        self.priors_ = self.model.priors_
        self.covariance_ = self.model.covariance_

    def _predict(self, X):
        return self.model.predict(X)

    def fit_transform(self, X, y=None):
        return self.model.fit_transform(X, y)

    def transform(self, X):
        return self.model.transform(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def score(self, X, y):
        return self.model.score(X, y)

    def plot2D(self, X, y):
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

        ax.set_xlabel("LD1")
        ax.set_ylabel("LD2")
        ax.set_title("LDA Decision Boundary")
        ax.legend(loc="best")
        plt.show()
