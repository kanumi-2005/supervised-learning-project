import numpy as np
from scipy.stats import norm
from sklearn.base import ClassifierMixin
from sklearn.metrics import log_loss
from ..base.basegdmodel import BaseGDModel


def normal_pdf(x):
    return norm.pdf(x)

def normal_cdf(x):
    return norm.cdf(x)


class ProbitClassifier(ClassifierMixin, BaseGDModel):

    def __init__(
        self,
        lr=0.01,
        max_iter=50,
        batch_size=1024,
        lr_sched="step_decay",
        step_size=10,
        decay_factor=0.5,
        random_state=42,
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
        self.step_size = step_size
        self.decay_factor = decay_factor

    def _encode_labels(self, y):
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("Probit only supports binary classification")
        return (y == self.classes_[1]).astype(float)

    def _init_params(self, X, y):
        n_features = X.shape[1]
        self.w = np.zeros(n_features + 1)
        self.y_bin_ = self._encode_labels(y)

    def _loss(self, X, y):
        probs = self.predict_proba(X)
        return log_loss(y, probs)

    def _grad(self, X, y):
        n_samples = X.shape[0]
        X_design = np.c_[np.ones(n_samples), X]

        y_bin = (y == self.classes_[1]).astype(float)

        z = X_design @ self.w
        p = normal_cdf(z)
        pdf = normal_pdf(z)

        eps = 1e-10
        p = np.clip(p, eps, 1 - eps)

        # gradient of negative log-likelihood
        grad = ((p - y_bin) * pdf / (p * (1 - p))) @ X_design
        return grad / n_samples

    def _update_params(self, grad, iteration):
        self.w -= self._lr(iteration) * grad

    def _lr(self, iteration):
        if self.lr_sched is None:
            return self.lr
        elif self.lr_sched == "step_decay":
            return self.lr * (self.decay_factor ** \
                              (iteration // self.step_size))
        else:
            raise ValueError(f"Unknown lr_sched: {self.lr_sched}")

    def _extra_logs(self, X, y, grad, iter):
        return {"lr": self._lr(iter)}

    def _predict(self, X):
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]

    def predict_proba(self, X):
        n_samples = X.shape[0]
        X_design = np.c_[np.ones(n_samples), X]

        z = X_design @ self.w
        p = normal_cdf(z)

        return np.c_[1 - p, p]

    @property
    def intercept_(self):
        return self.w[0]

    @property
    def coef_(self):
        return self.w[1:]