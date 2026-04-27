import numpy as np


def fisher_ratio(X, y):
    """
    Compute Fisher score (Fisher ratio) for each feature.

    The Fisher ratio measures the discriminative power of each feature
    by comparing between-class variance to within-class variance.
    Higher values indicate better class separability.

    For each feature j, the Fisher ratio is defined as:

        J_j = sum_c n_c (mu_cj - mu_j)^2 / sum_c n_c sigma_cj^2

    where mu_cj and sigma_cj^2 are the mean and variance of feature j
    for class c, and mu_j is the global mean of feature j.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input data matrix.

    y : array-like of shape (n_samples,)
        Target class labels.

    Returns
    -------
    J : ndarray of shape (n_features,)
        Fisher scores for each feature.

    Notes
    -----
    This metric is commonly used for feature selection in
    classification problems. It assumes numerical features and
    discrete class labels.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[1, 2], [2, 3], [10, 20], [11, 21]])
    >>> y = np.array([0, 0, 1, 1])
    >>> fisher_ratio(X, y)
    array([..., ...])
    """
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
