import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, clone
from sklearn.utils import resample, check_random_state


class BiasVarianceAnalyzer(BaseEstimator):
    def __init__(self, estimator, n_bootstrap=200, random_state=42):
        self.estimator = estimator
        self.n_bootstrap = n_bootstrap
        self.random_state = random_state

    def fit(self, X, y, lambdas=None):
        self.X_ = X
        self.y_ = y

        self.lambdas_ = (
            np.logspace(-10, 1, 12)
            if lambdas is None
            else np.asarray(lambdas)
        )

        rng = check_random_state(self.random_state)

        self.bias2_ = []
        self.variance_ = []

        for l in self.lambdas_:
            model = clone(self.estimator)
            if hasattr(model, "alpha"):
                model.alpha = l

            predictions = []

            for _ in range(self.n_bootstrap):
                X_s, y_s = resample(X, y, random_state=rng)
                m = clone(model)
                m.fit(X_s, y_s)
                predictions.append(m.predict(X))

            predictions = np.array(predictions)

            mean_pred = np.mean(predictions, axis=0)

            bias2 = np.mean((mean_pred - y) ** 2)
            variance = np.mean(np.var(predictions, axis=0, ddof=1))

            self.bias2_.append(bias2)
            self.variance_.append(variance)

        self.bias2_ = np.array(self.bias2_)
        self.variance_ = np.array(self.variance_)

        return self

    def plot(self):
        fig, ax = plt.subplots(layout="constrained")

        ax.plot(self.lambdas_, self.bias2_, label="Bias²")
        ax.plot(self.lambdas_, self.variance_, label="Variance")

        ax.set_xscale("log")
        ax.set_xlabel(r"$log_{10}(\lambda)$")
        ax.set_ylabel("Error")
        ax.set_title("Bias–Variance vs Regularization Path")
        ax.legend(loc="best")

        return fig, ax

    def get_bias_variance(self):
        return self.bias2_, self.variance_
