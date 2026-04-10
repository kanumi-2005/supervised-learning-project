from .basemodel import BaseModel
import numpy as np


class BaseGDModel(BaseModel):
    def __init__(
            self,
            lr,
            max_iter,
            store_history,
            batch_size=None,
            random_state=42
        ):
        super().__init__()
        self.lr = lr
        self.max_iter = max_iter
        self.store_history = store_history
        self.batch_size = batch_size
        self.random_state = random_state
        self.rng = np.random.default_rng(self.random_state)

    def _fit(self, X, y, **kwargs):
        X_val = kwargs.get("X_val", None)
        y_val = kwargs.get("y_val", None)

        self._init_params(X, y)
        self.n_samples = X.shape[0]

        if self.store_history:
            self.train_loss_history_ = []
            self.val_loss_history_ = []

        batch_size = self.batch_size or self.n_samples

        for it in range(self.max_iter):
            indices = self.rng.permutation(self.n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for start in range(0, self.n_samples, batch_size):
                end = start + batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                grad = self._grad(X_batch, y_batch)
                self._update_params(grad, iteration=it)

            train_loss = self._loss(X, y)

            val_loss = None
            if X_val is not None and y_val is not None:
                val_loss = self._loss(X_val, y_val)

            if self.store_history:
                self.train_loss_history_.append(train_loss)
                if val_loss is not None:
                    self.val_loss_history_.append(val_loss)

            base_log = {
                "iter": it,
                "train_loss": train_loss,
                "val_loss": val_loss
            }

            extra_log = self._extra_logs(
                X=X,
                y=y,
                grad=None,
                iter=it
            )

            if extra_log:
                base_log.update(extra_log)

            yield base_log

        self.n_iter_ = self.max_iter

    def _init_params(self, X, y):
        raise NotImplementedError

    def _loss(self, X, y):
        raise NotImplementedError

    def _grad(self, X, y):
        raise NotImplementedError

    def _update_params(self, grad, iteration):
        raise NotImplementedError

    def _extra_logs(self, X, y, grad, iter):
        return {}
