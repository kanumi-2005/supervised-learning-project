from sklearn.linear_model import LassoCV
from sklearn.feature_selection import SequentialFeatureSelector


def forward_selection(X, y, estimator, k_features):
    sfs = SequentialFeatureSelector(
        estimator=estimator,
        n_features_to_select=k_features,
        direction='forward'
    )

    sfs.fit(X, y)
    selected_features = X.columns[sfs.get_support()]

    return list(selected_features)

def backward_elimination(X, y, estimator, k_features):
    sfs = SequentialFeatureSelector(
        estimator=estimator,
        n_features_to_select=k_features,
        direction='backward'
    )

    sfs.fit(X, y)
    selected_features = X.columns[sfs.get_support()]

    return list(selected_features)

def lasso_selection(X, y, alphas=(0.1, 1.0, 10.0), cv=10):
    lasso = LassoCV(alphas=alphas, cv=cv)
    lasso.fit(X, y)

    selected_features = X.columns[lasso.coef_ != 0]

    return list(selected_features)
