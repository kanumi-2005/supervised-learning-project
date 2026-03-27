import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


class MBGD(RegressorMixin, BaseEstimator):
    def __init__(
        self,
        batch_size = 4,
        lr = 0.01,
        epochs = 100,
        lr_sched = None,
        seed = 42,
        eta_max = 0.01,
        eta_min = 0.0001,
        step_size = 50,
        decay_factor = 0.5
    ):
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.lr_sched = lr_sched
        self.seed = seed
        self.eta_max = eta_max
        self.eta_min = eta_min
        self.step_size = step_size
        self.decay_factor = decay_factor

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)

        n_samples, n_features = X.shape
        X_design = np.c_[np.ones(n_samples), X]
        w = np.zeros(n_features + 1)
        for epoch in range(self.epochs):
            indices = rng.permutation(n_samples)
            X_shuffled = X_design[indices]
            y_shuffled = y[indices]

            for start in range(0, n_samples, self.batch_size):
                end = start + self.batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                grad = - X_batch.T @ (y_batch - X_batch @ w) / len(X_batch)
                w -= self._lr(epoch) * grad

        self.intercept_ = w[0]
        self.coef_ = w[1:]
        return self

    def predict(self, X):
        return self.intercept_ + X @ self.coef_

    def _lr(self, epoch):
        if self.lr_sched is None:
            return self.lr
        elif self.lr_sched == "step_decay":
            return self.lr * (self.decay_factor ** (epoch // self.step_size))
        elif self.lr_sched == "cosine_annealing":
            return self.eta_min + 0.5 * (self.eta_max - self.eta_min) * \
                   (1 + np.cos(np.pi * epoch / self.epochs))
        else:
            raise ValueError(f"Unknown lr_sched: {self.lr_sched}")


if __name__ == "__main__":
    from ..dataset import CaliforniaHousingDataset as Dataset
    from ..pipeline import get_pipeline

    d = Dataset()
    d.split()

    model = get_pipeline(MBGD())
    model.fit(d.X_train, d.y_train)
    y_pred = model.predict(d.X_test)
    y_true = d.y_test

    sse = np.sum(np.square(y_true - y_pred))
    mse = np.mean((y_true - y_pred) ** 2)
    print(f"MSE = {mse:.4f}")
