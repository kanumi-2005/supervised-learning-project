import numpy as np
import time
from sklearn.base import ClassifierMixin
from sklearn.metrics import log_loss
from ..base.basemodel import BaseModel


def sigmoid(x):
    x = np.clip(x, -50, 50)
    return 1 / (1 + np.exp(-x))


class IRLSClassifier(ClassifierMixin, BaseModel):
    def __init__(self, max_iter=20, store_history=True):
        self.max_iter = max_iter
        self.store_history = store_history

    def _encode_labels(self, y):
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("IRLSClassifier only supports binary")
        return (y == self.classes_[1]).astype(float)

    def _init_params(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features + 1)
        self.y_bin_ = self._encode_labels(y)

    def _loss(self, X, y):
        probs = self.predict_proba(X)
        return log_loss(y, probs)

    def _fit(self, X, y, **kwargs):
        X_val = kwargs.get("X_val", None)
        y_val = kwargs.get("y_val", None)

        self._init_params(X, y)

        n_samples = X.shape[0]
        X_design = np.c_[np.ones(n_samples), X]

        if self.store_history:
            self.train_loss_history_ = []
            self.val_loss_history_ = []
            self.time_history_ = []

        start_time = time.perf_counter()

        for it in range(self.max_iter):
            z = X_design @ self.w
            p = sigmoid(z)

            W = p * (1 - p)
            H = X_design.T @ (W[:, None] * X_design)
            g = X_design.T @ (p - self.y_bin_)

            try:
                delta = np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                delta = np.linalg.pinv(H) @ g

            self.w -= delta

            train_loss = self._loss(X, y)

            val_loss = None
            if X_val is not None and y_val is not None:
                val_loss = self._loss(X_val, y_val)

            elapsed = time.perf_counter() - start_time

            if self.store_history:
                self.train_loss_history_.append(train_loss)
                if val_loss is not None:
                    self.val_loss_history_.append(val_loss)
                self.time_history_.append(elapsed)

            yield {
                "iter": it,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "time": elapsed
            }

        self.n_iter_ = self.max_iter

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
    model = IRLSClassifier()

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
