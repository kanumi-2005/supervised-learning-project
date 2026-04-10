import numpy as np
from sklearn.base import RegressorMixin
from sklearn.metrics import mean_squared_error
from ..base.basegdmodel import BaseGDModel


class MBGD(RegressorMixin, BaseGDModel):
    def __init__(
        self,
        batch_size=64,
        lr=0.01,
        max_iter=150,
        lr_sched=None,
        random_state=42,
        min_lr=0.0001,
        step_size=30,
        decay_factor=0.5,
        store_history=True
    ):
        super().__init__(
            lr=lr,
            max_iter=max_iter,
            store_history=store_history,
            batch_size=batch_size,
            random_state=random_state
        )
        self.lr_sched = lr_sched
        self.min_lr = min_lr
        self.step_size = step_size
        self.decay_factor = decay_factor

    def _init_params(self, X, y):
        n_features = X.shape[1]
        self.w = np.zeros(n_features + 1)

    def _loss(self, X, y):
        y_pred = self.w[0] + X @ self.w[1:]
        return mean_squared_error(y, y_pred)

    def _grad(self, X, y):
        n_samples = X.shape[0]
        X_design = np.c_[np.ones(n_samples), X]
        return - X_design.T @ (y - X_design @ self.w) / n_samples

    def _update_params(self, grad, iteration):
        self.w -= self._lr(iteration) * grad

    def _lr(self, iteration):
        if self.lr_sched is None:
            return self.lr
        elif self.lr_sched == "step_decay":
            return self.lr * (self.decay_factor ** (iteration // self.step_size))
        elif self.lr_sched == "cosine_annealing":
            return self.min_lr + 0.5 * (self.lr - self.min_lr) * \
                   (1 + np.cos(np.pi * iteration / self.max_iter))
        else:
            raise ValueError(f"Unknown lr_sched: {self.lr_sched}")

    def _extra_logs(self, X, y, grad, iter):
        return {"lr": self._lr(iter)}

    def _predict(self, X):
        return self.w[0] + X @ self.w[1:]

    @property
    def intercept_(self):
        return self.w[0]

    @property
    def coef_(self):
        return self.w[1:]


if __name__ == "__main__":
    from ..dataset import CaliforniaHousingDataset as Dataset
    from ..pipeline import get_pipeline
    from sklearn.metrics import mean_squared_error

    d = Dataset()
    d.split()

    model = get_pipeline(MBGD(lr_sched="step_decay"))
    model.fit(d.X_train, d.y_train)
    y_pred = model.predict(d.X_test)
    y_true = d.y_test

    mse = mean_squared_error(y_true, y_pred)

    print(f"MSE = {mse:.4f}")
