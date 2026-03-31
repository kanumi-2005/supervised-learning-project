from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


class QDA(ClassifierMixin, BaseEstimator):
    def __init__(self):
        self.model = QuadraticDiscriminantAnalysis(store_covariance=True)

    def fit(self, X, y):
        self.model.fit(X, y)
        self.means_ = self.model.means_
        self.priors_ = self.model.priors_
        self.covariances_ = self.model.covariance_
        return self

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        return self.model.score(X, y)
