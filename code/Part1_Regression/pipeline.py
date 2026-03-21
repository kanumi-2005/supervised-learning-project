from numpy import transpose
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline


def get_pipeline(predictor, features = None):
    pipe = Pipeline(
        steps=[
            ("features", features),
            ("scaler", MinMaxScaler()),
            ("predictor", predictor)
        ]
    )

    return pipe
