from numpy import transpose
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline


def get_pipeline(predictor, features = None):
    pipe = Pipeline(
        steps=[
            ("features", features),
            ("scaler", StandardScaler()),
            ("predictor", predictor)
        ]
    )

    return pipe
