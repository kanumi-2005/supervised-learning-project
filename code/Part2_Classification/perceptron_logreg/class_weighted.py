import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.metrics import log_loss
from ..base.basegdmodel import BaseGDModel


def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


class WeightedSoftmaxClassifier(ClassifierMixin, BaseGDModel):
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

    # ENCODE LABEL
    def _encode_labels(self, y):
        self.classes_ = np.unique(y)
        self.class_to_index_ = {c: i for i, c in enumerate(self.classes_)}
        return np.array([self.class_to_index_[c] for c in y])

    def _one_hot(self, y, n_classes):
        one_hot = np.zeros((len(y), n_classes))
        one_hot[np.arange(len(y)), y] = 1
        return one_hot

    # INIT PARAMS
    def _init_params(self, X, y):
        n_samples, n_features = X.shape

        y_encoded = self._encode_labels(y)
        self.n_classes_ = len(self.classes_)

        self.X_ = X
        self.y_encoded_ = y_encoded

        # ===== CLASS WEIGHT =====
        counts = np.bincount(y_encoded, minlength=self.n_classes_)
        self.class_weights_ = len(y_encoded) / (counts + 1e-8)

        self.W = np.zeros((n_features + 1, self.n_classes_))

    # LOSS
    def _loss(self, X, y):
        probs = self.predict_proba(X)
        return log_loss(y, probs, labels=self.classes_)

    # GRADIENT (FIXED WEIGHTED)
    def _grad(self, X, y):
        n_samples = X.shape[0]
        X_design = np.c_[np.ones(n_samples), X]

        # encode y
        y_encoded = np.array([self.class_to_index_[c] for c in y])
        Y = self._one_hot(y_encoded, self.n_classes_)

        probs = softmax(X_design @ self.W)

        # ===== SAMPLE WEIGHTS =====
        sample_weights = self.class_weights_[y_encoded]

        error = (probs - Y)
        error *= sample_weights[:, None]

        grad = (X_design.T @ error) / n_samples
        return grad

    # UPDATE
    def _update_params(self, grad, iteration):
        self.W -= self._lr(iteration) * grad

    # LR SCHEDULE
    def _lr(self, iteration):
        if self.lr_sched is None:
            return self.lr
        elif self.lr_sched == "step_decay":
            return self.lr * (self.decay_factor ** (iteration // self.step_size))
        else:
            raise ValueError(f"Unknown lr_sched: {self.lr_sched}")

    def _extra_logs(self, X, y, grad, iter):
        return {"lr": self._lr(iter)}

    def _predict(self, X):
        probs = self.predict_proba(X)
        indices = np.argmax(probs, axis=1)
        return self.classes_[indices]
    # PREDICT PROBA 
    def predict_proba(self, X):
        n_samples = X.shape[0]
        X_design = np.c_[np.ones(n_samples), X]
        return softmax(X_design @ self.W)

    # ACCESSORS
    @property
    def intercept_(self):
        return self.W[0]

    @property
    def coef_(self):
        return self.W[1:]