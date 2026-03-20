from numpy import transpose
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline


def get_pipeline(regressor):
    """
    Create a regression pipeline with feature scaling and target transformation.

    This pipeline performs:
    - Min-max scaling on input features (X)
    - Min-max scaling on the target variable (y) using TransformedTargetRegressor
    - Applies the provided regression model

    Parameters
    ----------
    regressor : object
        A scikit-learn compatible regressor implementing `fit` and `predict`,
        e.g., LinearRegression, Ridge, etc.

    Returns
    -------
    sklearn.pipeline.Pipeline
        A pipeline consisting of:
        - "scaler": MinMaxScaler applied to input features
        - "predictor": TransformedTargetRegressor wrapping the given regressor
          and applying MinMaxScaler to the target variable

    Notes
    -----
    - The target variable (y) is scaled during training and inverse-transformed
      during prediction.
    - This is useful when the scale of y affects model performance, especially
      for regularized models.

    Examples
    --------
    >>> from sklearn.linear_model import LinearRegression
    >>> pipe = get_pipeline(LinearRegression())
    >>> pipe.fit(X_train, y_train)
    >>> y_pred = pipe.predict(X_test)
    """
    pipe = Pipeline(
        steps=[
            ("scaler", MinMaxScaler()),
            (
                "predictor",
                TransformedTargetRegressor(
                    regressor=regressor,
                    transformer=MinMaxScaler()
                )
            )
        ]
    )

    return pipe
