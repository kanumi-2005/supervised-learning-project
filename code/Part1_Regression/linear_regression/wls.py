import numpy as np
from sklearn.base import RegressorMixin
from ..base.basemodel import BaseModel


class WLS(RegressorMixin, BaseModel):
    def __init__(self, weights):
        self.weights = weights

    def _fit(self, X, y, **kwargs):
        X_design = np.c_[np.ones(X.shape[0]), X]

        X_weighted = X_design.T * self.weights
        A = X_weighted @ X_design
        b = X_weighted @ y
        w = np.linalg.pinv(A) @ b

        self.intercept_ = w[0]
        self.coef_ = w[1:]

    def _predict(self, X):
        return X @ self.coef_ + self.intercept_


if __name__ == "__main__":
    from ..dataset import CaliforniaHousingDataset as Dataset
    from ..pipeline import get_pipeline
    from sklearn.metrics import mean_squared_error

    d = Dataset()
    d.split()

    model = get_pipeline(WLS(np.ones(d.train_size())))
    model.fit(d.X_train, d.y_train)
    y_pred = model.predict(d.X_test)
    y_true = d.y_test

    mse = mean_squared_error(y_true, y_pred)

    print(f"MSE = {mse:.4f}")
