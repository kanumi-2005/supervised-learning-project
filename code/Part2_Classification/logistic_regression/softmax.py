import numpy as np
from sklearn.base import ClassifierMixin, BaseEstimator


def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


class SoftmaxClassifier(ClassifierMixin, BaseEstimator):

    def __init__(
        self,
        lr=0.01,
        max_iter=150,
        batch_size=64,
        lr_sched="step_decay",
        step_size=50,
        decay_factor=0.5,
        random_state=42
    ):
        self.lr = lr
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.lr_sched = lr_sched
        self.step_size = step_size
        self.decay_factor = decay_factor
        self.random_state = random_state

    def _encode_labels(self, y):
        self.classes_ = np.unique(y)
        self.class_to_index_ = {c: i for i, c in enumerate(self.classes_)}
        return np.array([self.class_to_index_[c] for c in y])

    def _one_hot(self, y, n_classes):
        one_hot = np.zeros((len(y), n_classes))
        one_hot[np.arange(len(y)), y] = 1
        return one_hot

    def _lr(self, iteration):
        if self.lr_sched is None:
            return self.lr
        elif self.lr_sched == "step_decay":
            return self.lr * (self.decay_factor ** \
                   (iteration // self.step_size))
        else:
            raise ValueError(f"Unknown lr_sched: {self.lr_sched}")

    def fit(self, X, y):
        rng = np.random.default_rng(self.random_state)

        n_samples, n_features = X.shape

        y_encoded = self._encode_labels(y)
        n_classes = len(self.classes_)

        X_design = np.c_[np.ones(n_samples), X]

        W = np.zeros((n_features + 1, n_classes))

        for i in range(self.max_iter):
            indices = rng.permutation(n_samples)
            X_shuffled = X_design[indices]
            y_shuffled = y_encoded[indices]

            for start in range(0, n_samples, self.batch_size):
                end = start + self.batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                y_pred_proba = softmax(X_batch @ W)
                Y = self._one_hot(y_batch, n_classes)

                grad = (X_batch.T @ (y_pred_proba - Y)) / X_batch.shape[0]
                W -= self._lr(i) * grad

        self.intercept_ = W[0]
        self.coef_ = W[1:]
        return self

    def predict_proba(self, X):
        scores = self.intercept_ + X @ self.coef_
        return softmax(scores)

    def predict(self, X):
        probs = self.predict_proba(X)
        indices = np.argmax(probs, axis=1)
        return self.classes_[indices]


if __name__ == "__main__":
    from ..dataset import CovtypeDataset as Dataset
    from sklearn.metrics import accuracy_score

    d = Dataset()
    d.split()

    model = SoftmaxClassifier()

    model.fit(d.X_train, d.y_train)
    y_pred = model.predict(d.X_test)
    y_true = d.y_test

    acc = accuracy_score(y_true, y_pred)
    print(f"Acc = {acc:.4f}")
