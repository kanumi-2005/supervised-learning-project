import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error


class LassoRegression(BaseEstimator, RegressorMixin):
    def __init__(
            self,
            alpha=1.0,
            learning_rate=0.001,
            max_iter=1000,
            warm_start=False,
        ):

        self.alpha = alpha
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.warm_start = warm_start

    def fit(self, X, y):
        n_samples, n_features = X.shape

        if not (self.warm_start and hasattr(self, "coef_")):
            self.coef_ = np.zeros(n_features)
            self.intercept_ = 0.0

        for _ in range(self.max_iter):
            y_pred = X @ self.coef_ + self.intercept_
            errors = y_pred - y


            grad_w = (2.0 / n_samples) * (X.T @ errors) \
                + self.alpha * np.sign(self.coef_)
            grad_b = (2.0 / n_samples) * np.sum(errors)

            self.coef_ -= self.learning_rate * grad_w
            self.intercept_ -= self.learning_rate * grad_b

        self.n_features_ = n_features
        return self

    def predict(self, X):
        return X @ self.coef_ + self.intercept_


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

        fig, ax = plt.subplots(layout="constrained")

        for i in range(coefs.shape[1]):
            ax.plot(np.log10(alphas), coefs[:, i], label=f'Feature {i+1}')

        ax.set_title(title)
        ax.set_xlabel(r'$\log_{10}(\lambda)$')
        ax.set_ylabel('Coefficients')
        ax.grid(True)
        ax.legend(loc='best', fontsize='small')

        plt.show()
