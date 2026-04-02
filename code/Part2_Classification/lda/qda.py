from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


class QDA(ClassifierMixin, BaseEstimator):
    def __init__(self, reg_param=0.001):
        self.reg_param = reg_param
        self.model = QuadraticDiscriminantAnalysis(
            reg_param=reg_param,
            store_covariance=True
        )

    def fit(self, X, y):
        self.model.fit(X, y)
        self.classes_ = self.model.classes_
        self.means_ = self.model.means_
        self.priors_ = self.model.priors_
        self.covariances_ = self.model.covariance_
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def score(self, X, y):
        return self.model.score(X, y)
