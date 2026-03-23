import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.linear_model import SGDRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error


class LassoRegression(SGDRegressor):
    def __init__(self, alpha=0.0001, warm_start=False, random_state=42):
        super().__init__(
            loss='squared_error',
            penalty='l1',
            alpha=alpha,
            warm_start=warm_start,
            random_state=random_state
        )


class LassoRegressionCV(BaseEstimator, RegressorMixin):
    def __init__(self, alphas=(0.1, 1.0, 10.0), cv=10, random_state=42):
        self.alphas = alphas
        self.cv = cv
        self.random_state = random_state

    def fit(self, X, y):
        X, y = np.array(X), np.array(y).ravel()
        self.alphas = sorted(self.alphas, reverse=True)
        kf = KFold(
            n_splits=self.cv,
            shuffle=True,
            random_state=self.random_state
        )
        mse_scores = np.zeros(len(self.alphas))
        coefs_folds = []

        for train_idx, val_idx in kf.split(X):
            fold_model = LassoRegression(
                warm_start=True,
                random_state=self.random_state
            )
            fold_coefs = []

            for i, alpha in enumerate(self.alphas):
                fold_model.set_params(alpha=alpha)
                fold_model.fit(X[train_idx], y[train_idx])

                pred = fold_model.predict(X[val_idx])
                mse_scores[i] += mean_squared_error(y[val_idx], pred)

                fold_coefs.append(fold_model.coef_.copy())

            coefs_folds.append(fold_coefs)

        avg_mse = mse_scores / self.cv
        best_idx = np.argmin(avg_mse)
        self.best_alpha_ = self.alphas[best_idx]
        self.result_ = avg_mse

        coefs_folds = np.array(coefs_folds)
        self.coefs_path_ = np.mean(coefs_folds, axis=0)

        self.final_model_ = LassoRegression(
            alpha=self.best_alpha_,
            warm_start=False
        )
        self.final_model_.fit(X, y)
        self.coef_ = self.final_model_.coef_
        self.intercept_ = self.final_model_.intercept_

        return self

    def predict(self, X):
        return self.final_model_.predict(X)

    def plot_regularization_path(self, title=None):
        if self.coefs_path_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        alphas = np.array(self.alphas)
        coefs = self.coefs_path_

        plt.figure(figsize=(12, 6))
        for i in range(coefs.shape[1]):
            plt.plot(np.log10(alphas), coefs[:, i], label=f'Feature {i+1}')

        plt.title(title)
        plt.xlabel(r'$\log_{10}(\lambda)$')
        plt.ylabel('Coefficients')
        plt.grid(True)
        plt.legend(loc='best', fontsize='small')
        plt.tight_layout()
        plt.show()
