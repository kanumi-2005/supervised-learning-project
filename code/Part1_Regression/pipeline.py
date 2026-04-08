from numpy import transpose
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline


def get_pipeline(predictor, features=None):
    steps = []
    steps.append(("scaler", StandardScaler()))
    if features is not None:
        steps.append(("features", features))
    steps.append(("predictor", predictor))
    return Pipeline(steps)
