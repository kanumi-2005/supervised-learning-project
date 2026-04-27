from numpy import transpose
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline


def get_pipeline(predictor, features=None, use_interaction=False):
    """
    Construct a scikit-learn Pipeline for regression models.

    This function builds a modular preprocessing and modeling pipeline
    that optionally includes interaction feature generation, feature
    scaling, optional feature selection/transformer, and a final
    predictor model.

    The pipeline is compatible with scikit-learn estimators and is
    intended to standardize preprocessing steps before fitting a
    regression model.

    Parameters
    ----------
    predictor : estimator object
        A regression model implementing fit/predict interface.

    features : transformer object, default=None
        Optional feature transformer or selector applied after scaling.

    use_interaction : bool, default=False
        If True, includes an interaction feature expansion step using
        InteractionBasis from nonlinear_basis.interaction.

    Returns
    -------
    pipeline : Pipeline
        A scikit-learn Pipeline object consisting of the following steps:

        - interaction (optional): InteractionBasis feature expansion
        - scaler: StandardScaler normalization
        - features (optional): user-defined feature transformer
        - predictor: final regression estimator

    Notes
    -----
    The interaction step is only included if use_interaction is True.
    StandardScaler is always applied before the predictor.

    Examples
    --------
    >>> from sklearn.linear_model import LinearRegression
    >>> pipe = get_pipeline(LinearRegression(), use_interaction=True)
    >>> pipe.fit(X, y)
    >>> pipe.predict(X)
    """

    steps = []
    if use_interaction:
        from nonlinear_basis.interaction import InteractionBasis
        steps.append(("interaction", InteractionBasis()))

    steps.append(("scaler", StandardScaler()))
    if features is not None:
        steps.append(("features", features))
    steps.append(("predictor", predictor))
    return Pipeline(steps)
