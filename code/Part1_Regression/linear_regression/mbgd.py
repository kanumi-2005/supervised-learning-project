import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


class MBGD(RegressorMixin, BaseEstimator):
    def __init__(
        self,
        batch_size = 4,
        lr = 0.01,
        max_iter = 100,
        lr_sched = None,
        random_state = 42,
        min_lr = 0.0001,
        step_size = 50,
        decay_factor = 0.5
    ):
        self.batch_size = batch_size
        self.lr = lr
        self.max_iter = max_iter
        self.lr_sched = lr_sched
        self.random_state = random_state
        self.min_lr = min_lr
        self.step_size = step_size
        self.decay_factor = decay_factor

    def fit(self, X, y):
        rng = np.random.default_rng(self.random_state)

        n_samples, n_features = X.shape
        X_design = np.c_[np.ones(n_samples), X]
        w = np.zeros(n_features + 1)
        for i in range(self.max_iter):
            indices = rng.permutation(n_samples)
            X_shuffled = X_design[indices]
            y_shuffled = y[indices]

            for start in range(0, n_samples, self.batch_size):
                end = start + self.batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                grad = - X_batch.T @ (y_batch - X_batch @ w) / len(X_batch)
                w -= self._lr(i) * grad

        self.intercept_ = w[0]
        self.coef_ = w[1:]
        return self

    def predict(self, X):
        return self.intercept_ + X @ self.coef_

    def _lr(self, iteration):
        if self.lr_sched is None:
            return self.lr
        elif self.lr_sched == "step_decay":
            return self.lr * (self.decay_factor ** \
                   (iteration // self.step_size))
        elif self.lr_sched == "cosine_annealing":
            return self.min_lr + 0.5 * (self.lr - self.min_lr) * \
                   (1 + np.cos(np.pi * iteration / self.max_iter))
        else:
            raise ValueError(f"Unknown lr_sched: {self.lr_sched}")


if __name__ == "__main__":
    from ..dataset import CaliforniaHousingDataset as Dataset
    from ..pipeline import get_pipeline
    from sklearn.metrics import mean_squared_error

    d = Dataset()
    d.split()

    model = get_pipeline(MBGD())
    model.fit(d.X_train, d.y_train)
    y_pred = model.predict(d.X_test)
    y_true = d.y_test

    mse = mean_squared_error(y_true, y_pred)

    print(f"MSE = {mse:.4f}")
