import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error


class ElasticNet(BaseEstimator, RegressorMixin):
    def __init__(
            self,
            alpha_1=1.0,
            alpha_2=0.5,
            learning_rate=0.001,
            max_iter=1000
        ):

        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.learning_rate = learning_rate
        self.max_iter = max_iter

    def fit(self, X, y):
        n_samples, n_features = X.shape
        y = np.asarray(y).ravel()

        self.coef_ = np.zeros(n_features)
        self.intercept_ = 0.0

        for _ in range(self.max_iter):
            y_pred = X @ self.coef_ + self.intercept_
            errors = y_pred - y

            grad_mse = (2.0 / n_samples) * (X.T @ errors)
            grad_l2 = 2.0 * self.alpha_2 * self.coef_
            grad_l1 = self.alpha_1 * np.sign(self.coef_)

            grad_w = grad_mse + grad_l2 + grad_l1
            grad_b = (2.0 / n_samples) * np.sum(errors)

            self.coef_ -= self.learning_rate * grad_w
            self.intercept_ -= self.learning_rate * grad_b

        self.n_features_in_ = n_features
        return self

    def predict(self, X):
        return X @ self.coef_ + self.intercept_


class ElasticNetCV(BaseEstimator, RegressorMixin):
    def __init__(
            self,
            alpha_1s=(0.1, 1.0, 10.0),
            alpha_2s=(0.1, 1.0, 10.0),
            cv=10,
            random_state=42
        ):

        self.alpha_1s = alpha_1s
        self.alpha_2s = alpha_2s
        self.cv = cv
        self.random_state = random_state

    def fit(self, X, y):
        X, y = np.array(X), np.array(y).ravel()
        self.alpha_1s = sorted(self.alpha_1s, reverse=True)
        self.alpha_2s = sorted(self.alpha_2s, reverse=True)
        kf = KFold(
            n_splits=self.cv,
            shuffle=True,
            random_state=self.random_state
        )

        alpha_1_grid, alpha_2_grid = np.meshgrid(self.alpha_1s, self.alpha_2s)
        param_grid = np.vstack([alpha_1_grid.ravel(), alpha_2_grid.ravel()]).T
        mse_results = np.zeros(len(param_grid))

        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            fold_model = ElasticNet()

            for i, (alpha_1, alpha_2) in enumerate(param_grid):
                fold_model.set_params(alpha_1=alpha_1, alpha_2=alpha_2)
                fold_model.fit(X_train, y_train)

                pred = fold_model.predict(X_val)
                mse_results[i] += mean_squared_error(y_val, pred)

        avg_mse = mse_results / self.cv
        best_idx = np.argmin(avg_mse)
        self.best_alpha_1_, self.best_alpha_2_ = param_grid[best_idx]

        self.results_ = avg_mse.reshape(
            len(self.alpha_2s),
            len(self.alpha_1s)
        )

        self.final_model_ = ElasticNet(
            alpha_1=self.best_alpha_1_,
            alpha_2=self.best_alpha_2_
        )

        self.final_model_.fit(X, y)

        return self

    def predict(self, X):
        return self.final_model_.predict(X)

    def plot_optimal_region(self, title=None):
        plot_alpha_1s = np.sort(self.alpha_1s)
        plot_alpha_2s = np.sort(self.alpha_2s)

        log_alpha_1, log_alpha_2 = np.meshgrid(
            np.log10(plot_alpha_1s),
            np.log10(plot_alpha_2s)
        )

        flipped_results = np.flip(self.results_, axis=(0, 1))

        fig, ax = plt.subplots(layout="constrained")

        cp = ax.contourf(
            log_alpha_1,
            log_alpha_2,
            flipped_results,
            levels=20,
            cmap='viridis_r'
        )

        ax.plot(
            np.log10(self.best_alpha_1_),
            np.log10(self.best_alpha_2_),
            'ro',
            markersize=6,
            markeredgecolor='white',
            label='Optimal Point'
        )

        fig.colorbar(cp, ax=ax, label='Mean MSE')

        ax.set_xlabel(r'$\log_{10}(\lambda_1)$')
        ax.set_ylabel(r'$\log_{10}(\lambda_2)$')
        ax.set_title(title)
        ax.legend()

        plt.show()
