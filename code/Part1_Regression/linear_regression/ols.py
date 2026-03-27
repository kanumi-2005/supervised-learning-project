import numpy as np
from sklearn.base import RegressorMixin, BaseEstimator


class OLS(RegressorMixin, BaseEstimator):
    def fit(self, X: np.ndarray, y):
        X_design = np.c_[np.ones(len(X)), X]
        A = X_design.T @ X_design
        b = X_design.T @ y
        w = np.linalg.pinv(A) @ b
        self.intercept_ = w[0]
        self.coef_ = w[1:]

        return self

    def predict(self, X):
        return X @ self.coef_ + self.intercept_


if __name__ == "__main__":
    from ..dataset import CaliforniaHousingDataset as Dataset
    from ..pipeline import get_pipeline

    d = Dataset()
    d.split()

    model = get_pipeline(OLS())
    model.fit(d.X_train, d.y_train)
    y_pred = model.predict(d.X_test)
    y_true = d.y_test

    mse = np.mean(np.square(y_true - y_pred))

    print(f"MSE = {mse:.4f}")
