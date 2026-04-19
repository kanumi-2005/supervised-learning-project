from sklearn.linear_model import LassoCV
from sklearn.feature_selection import SequentialFeatureSelector


def forward_selection(X, y, estimator, k_features, feature_names):
    sfs = SequentialFeatureSelector(
        estimator=estimator,
        n_features_to_select=k_features,
        direction='forward'
    )

    sfs.fit(X, y)

    selected_idx = sfs.get_support(indices=True)
    selected_features = [feature_names[i] for i in selected_idx]

    return selected_features

def backward_elimination(X, y, estimator, k_features, feature_names):
    sfs = SequentialFeatureSelector(
        estimator=estimator,
        n_features_to_select=k_features,
        direction='backward'
    )

    sfs.fit(X, y)

    selected_idx = sfs.get_support(indices=True)
    selected_features = [feature_names[i] for i in selected_idx]

    return selected_features

def lasso_selection(X, y, feature_names, alphas=(0.1, 1.0, 10.0), cv=10):
    lasso = LassoCV(alphas=alphas, cv=cv)
    lasso.fit(X, y)

    selected_idx = [i for i, coef in enumerate(lasso.coef_) if abs(coef) > 1e-3]
    selected_features = [feature_names[i] for i in selected_idx]

    return selected_features
