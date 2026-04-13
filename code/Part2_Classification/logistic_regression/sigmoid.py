import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.metrics import log_loss
from ..base.basegdmodel import BaseGDModel


def sigmoid(x):
    x = np.clip(x, -50, 50)
    return 1 / (1 + np.exp(-x))


class SigmoidClassifier(ClassifierMixin, BaseGDModel):
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
            raise ValueError("SigmoidClassifier only supports binary")
        return (y == self.classes_[1]).astype(float)

    def _init_params(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features + 1)
        self.y_bin_ = self._encode_labels(y)

    def _loss(self, X, y):
        probs = self.predict_proba(X)
        return log_loss(y, probs)

    def _grad(self, X, y):
        n_samples = X.shape[0]
        X_design = np.c_[np.ones(n_samples), X]

        y_bin = (y == self.classes_[1]).astype(float)
        p = sigmoid(X_design @ self.w)

        return X_design.T @ (p - y_bin) / n_samples

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
        p = sigmoid(X_design @ self.w)
        return np.c_[1 - p, p]

    @property
    def intercept_(self):
        return self.w[0]

    @property
    def coef_(self):
        return self.w[1:]


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import StandardScaler

    # dataset
    X, y = make_classification(
        n_samples=500_000,
        n_features=10,
        n_informative=6,
        n_classes=2,
        random_state=42
    )

    # split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # model
    model = SigmoidClassifier(
        max_iter=20,
        step_size=4,
        batch_size=1024
    )

    # fit
    model.fit(X_train, y_train, X_val=X_val, y_val=y_val)

    # accuracy
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Acc = {acc:.4f}")

    # ===== PLOT =====
    epochs = range(len(model.train_loss_history_))

    # 1. loss vs epoch
    plt.figure(layout="constrained")
    plt.plot(epochs, model.train_loss_history_, label="train")
    plt.plot(epochs, model.val_loss_history_, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss vs Epoch")
    plt.legend()
    plt.show()

    # 2. loss vs time
    plt.figure(layout="constrained")
    plt.plot(model.time_history_, model.train_loss_history_, label="train")
    plt.plot(model.time_history_, model.val_loss_history_, label="val")
    plt.xlabel("Time (s)")
    plt.ylabel("Loss")
    plt.title("Loss vs Time")
    plt.legend()
    plt.show()
