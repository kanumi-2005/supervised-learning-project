import numpy as np


def fisher_ratio(X, y):
    classes = np.unique(y)
    n_features = X.shape[1]

    J = np.zeros(n_features)

    for j in range(n_features):
        num = 0
        den = 0

        mean_j = np.mean(X[:, j])

        for c in classes:
            X_cj = X[y == c, j]
            mean_cj = np.mean(X_cj)
            var_cj = np.var(X_cj, ddof=1)

            num += len(X_cj) * (mean_cj - mean_j) ** 2
            den += len(X_cj) * var_cj

        J[j] = num / den

    return J
