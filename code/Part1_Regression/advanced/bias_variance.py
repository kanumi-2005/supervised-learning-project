import numpy as np
from sklearn.base import clone
from sklearn.utils import resample, check_random_state


class BiasVarianceAnalyzer:
    def __init__(self, estimator, n_bootstrap=200, random_state=42):
        self.estimator = estimator
        self.n_bootstrap = n_bootstrap
        self.random_state = random_state

    def fit(self, X, y):
        rng = check_random_state(self.random_state)

        predictions = []

        for _ in range(self.n_bootstrap):
            X_sample, y_sample = resample(X, y, random_state=rng)
            model = clone(self.estimator)
            model.fit(X_sample, y_sample)
            y_pred = model.predict(X)
            predictions.append(y_pred)

        predictions = np.array(predictions)
        mean_pred = np.mean(predictions, axis=0)

        # bias^2
        self.bias2_ = np.mean((mean_pred - y) ** 2)

        # variance
        self.variance_ = np.mean(np.var(predictions, axis=0))

        return self
