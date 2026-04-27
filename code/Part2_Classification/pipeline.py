from numpy import transpose
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline


def get_pipeline(predictor, features=None):
    """
    Construct a machine learning pipeline with optional feature transform.

    This function builds a sklearn Pipeline consisting of an optional
    feature transformation step, standardization, and a final predictor.

    The pipeline ensures that input features are optionally transformed,
    then standardized using StandardScaler before being passed to the
    provided predictor model.

    Parameters
    ----------
    predictor : estimator object
        The final regression or prediction model that implements
        fit and predict methods.

    features : transformer or None, default=None
        Optional feature transformation step applied before scaling.
        If None, no feature transformation is applied.

    Returns
    -------
    pipeline : sklearn.pipeline.Pipeline
        A composed pipeline with steps:
        - features (optional transformation)
        - scaler (StandardScaler)
        - predictor (final estimator)

    Examples
    --------
    >>> from sklearn.linear_model import LinearRegression
    >>> model = get_pipeline(LinearRegression())
    >>> type(model)
    <class 'sklearn.pipeline.Pipeline'>

    >>> model.steps  # doctest: +SKIP
    [('features', None),
     ('scaler', StandardScaler()),
     ('predictor', LinearRegression())]
    """
    pipe = Pipeline(
        steps=[
            ("features", features),
            ("scaler", StandardScaler()),
            ("predictor", predictor),
        ]
    )

    return pipe
