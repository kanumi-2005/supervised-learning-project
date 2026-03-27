from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV


class KernelRidgeCV(BaseEstimator, RegressorMixin):
    def __init__(self, kernel='rbf', param_grid=None, cv=10):
        self.kernel = kernel
        self.param_grid = param_grid
        self.cv = cv

    def _default_param_grid(self):
        if self.kernel == 'rbf':
            return {
                "alpha": [0.1, 1, 10],
                "gamma": [0.01, 0.1, 1]
            }
        elif self.kernel == 'polynomial':
            return {
                "alpha": [0.1, 1, 10],
                "degree": [2, 3, 4],
                "coef0": [0, 1]
            }
        else:
            raise ValueError("Kernel must be 'rbf' or 'polynomial'")

    def fit(self, X, y):
        param_grid = self.param_grid if self.param_grid is not None \
            else self._default_param_grid()

        self.grid_ = GridSearchCV(
            KernelRidge(kernel=self.kernel),
            param_grid,
            cv=self.cv
        )

        self.grid_.fit(X, y)

        self.model_ = self.grid_.best_estimator_
        self.best_params_ = self.grid_.best_params_
        self.best_score_ = self.grid_.best_score_

        return self

    def predict(self, X):
        return self.model_.predict(X)

    def score(self, X, y):
        return self.model_.score(X, y)
