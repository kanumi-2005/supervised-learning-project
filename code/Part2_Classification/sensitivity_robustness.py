import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer, KNNImputer

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score
)

from sklearn.model_selection import train_test_split

METRIC_DIRECTION = {
    # classification (↑ good)
    "accuracy": 1,
    "precision_macro": 1,
    "recall_macro": 1,
    "f1_macro": 1,

    # regression
    "r2": 1,
    "mse": -1,
    "rmse": -1,
    "mae": -1,
}


def eval_classification(y_true, y_pred):
    """
    Compute classification metrics.

    Parameters
    ----------
    y_true : array-like
        Ground truth labels.

    y_pred : array-like
        Predicted labels.

    Returns
    -------
    dict
        Dictionary containing accuracy, macro precision,
        macro recall, and macro F1-score.

    Examples
    --------
    >>> eval_classification([0, 1], [0, 1])
    {'accuracy': 1.0, 'precision_macro': 1.0,
     'recall_macro': 1.0, 'f1_macro': 1.0}
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "recall_macro": recall_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def eval_regression(y_true, y_pred):
    """
    Compute regression metrics.

    Parameters
    ----------
    y_true : array-like
        Ground truth values.

    y_pred : array-like
        Predicted values.

    Returns
    -------
    dict
        Dictionary containing MSE, RMSE, MAE, and R2.

    Examples
    --------
    >>> eval_regression([1, 2], [1, 2])
    {'mse': 0.0, 'rmse': 0.0, 'mae': 0.0, 'r2': 1.0}
    """
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mse": mse,
        "rmse": np.sqrt(mse),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }


def build_pipeline(model, imputer=None, **params):
    """
    Build sklearn pipeline with optional imputation.

    Parameters
    ----------
    model : estimator
        Base model.

    imputer : str or None, default=None
        Imputation strategy: 'mean', 'median', 'knn', or None.

    **params :
        Parameters passed to the model.

    Returns
    -------
    Pipeline
        sklearn Pipeline with preprocessing and model.

    Examples
    --------
    >>> build_pipeline(model, imputer="mean")
    """
    if imputer == "mean":
        imp = SimpleImputer(strategy="mean")
    elif imputer == "median":
        imp = SimpleImputer(strategy="median")
    elif imputer == "knn":
        imp = KNNImputer(n_neighbors=5)
    elif imputer is None:
        imp = None
    else:
        raise ValueError(f"Unkown value imputer: {imputer}")

    model.set_params(**params)

    return Pipeline([
        ("imputer", imp),
        ("model", model)
    ])


def _run(pipe, dataset, task, **fit_params):
    """
    Train and evaluate pipeline on dataset.

    Parameters
    ----------
    pipe : Pipeline
        sklearn pipeline.

    dataset : object
        Dataset with train/test attributes.

    task : str or None
        'classification', 'regression', or None.

    **fit_params :
        Extra fit parameters.

    Returns
    -------
    dict or None
        Evaluation metrics or None if task is None.
    """
    X_train, y_train = dataset.X_train, dataset.y_train
    X_test, y_test = dataset.X_test, dataset.y_test

    pipe.fit(X_train, y_train, **fit_params)
    pred = pipe.predict(X_test)

    if task is None:
        return

    return (
        eval_classification(y_test, pred)
        if task == "classification"
        else eval_regression(y_test, pred)
    )


def sensitivity_split_analysis(models, dataset, task="classification",
                               n_runs=10, random_state=42):
    """
    Evaluate model sensitivity to train/test split variation.

    Parameters
    ----------
    models : dict
        Mapping model name to estimator.

    dataset : object
        Dataset with split method.

    task : str
        'classification' or 'regression'.

    n_runs : int
        Number of random splits per configuration.

    random_state : int
        Random seed.

    Returns
    -------
    pd.DataFrame
        Metrics over multiple splits.
    """
    results = []

    splits = {
        "60/40": 0.4,
        "70/30": 0.3,
        "80/20": 0.2
    }

    X_base = dataset.X.copy()
    y_base = dataset.y.copy()

    rng = np.random.default_rng(random_state)

    for split_name, test_size in splits.items():

        seeds = rng.integers(low=0, high=2**32 - 1, size=n_runs)

        for seed in seeds:

            dataset.X = X_base.copy()
            dataset.y = y_base.copy()

            dataset.split(
                train_size=1 - test_size,
                val_size=0.0,
                test_size=test_size,
                random_state=seed
            )

            for name, model in models.items():

                pipe = build_pipeline(clone(model))
                metrics = _run(pipe, dataset, task)

                results.append({
                    "model": name,
                    "split": split_name,
                    "seed": seed,
                    **metrics
                })

    return pd.DataFrame(results)


def _add_noise(X, sigma, rng):
    """
    Add Gaussian noise to features.

    Parameters
    ----------
    X : ndarray
        Input features.

    sigma : float
        Noise standard deviation.

    rng : Generator
        Random generator.

    Returns
    -------
    ndarray
        Noisy features.
    """
    X = X.copy()
    X += rng.normal(0, sigma, size=X.shape)
    return X


def noise_injection_analysis(models, dataset, sigmas=[0.1, 0.5, 1.0],
                             task="classification", random_state=42):
    """
    Evaluate model robustness under feature noise.

    Parameters
    ----------
    models : dict
        Models to evaluate.

    dataset : object
        Dataset object.

    sigmas : list
        Noise levels.

    task : str
        Problem type.

    random_state : int
        Seed.

    Returns
    -------
    pd.DataFrame
        Metrics and sensitivity scores.
    """
    results = []

    X_base = dataset.X.copy()

    dataset.X = X_base.copy()
    dataset.split(0.8, 0.0, 0.2)

    clean_metrics = {}

    for name, model in models.items():
        pipe = build_pipeline(clone(model))
        metrics = _run(pipe, dataset, task)
        clean_metrics[name] = metrics

        results.append({
            "model": name,
            "sigma": 0.0,
            **metrics,
            "is_clean": True
        })

    rng = np.random.default_rng(random_state)

    for sigma in sigmas:

        dataset.X = _add_noise(X_base, sigma, rng)
        dataset.split(0.8, 0.0, 0.2)

        for name, model in models.items():

            pipe = build_pipeline(clone(model))
            metrics = _run(pipe, dataset, task)

            base = {
                "model": name,
                "sigma": sigma,
                **metrics,
                "is_clean": False
            }

            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    clean_v = clean_metrics[name].get(k, None)

                    if clean_v is not None:
                        direction = METRIC_DIRECTION.get(k, 1)

                        sens_abs = direction * (clean_v - v)

                        sens_rel = (
                            sens_abs / abs(clean_v)
                            if abs(clean_v) > 1e-12 else None
                        )

                        base[f"{k}_sens_abs"] = sens_abs
                        base[f"{k}_sens_rel"] = sens_rel

            results.append(base)

    return pd.DataFrame(results)


def _corrupt(X, rate, rng):
    """
    Randomly mask features as missing values.

    Parameters
    ----------
    X : ndarray
        Input data.

    rate : float
        Missing rate.

    rng : Generator
        Random generator.

    Returns
    -------
    ndarray
        Corrupted dataset.
    """
    X = X.copy()
    mask = rng.random(X.shape) < rate
    X[mask] = np.nan
    return X


def feature_corruption_analysis(models, dataset,
                                missing_rates=[0.1, 0.2, 0.3],
                                imputers=["mean", "median", "knn"],
                                task="classification", random_state=42):
    """
    Evaluate model robustness under missing data.

    Parameters
    ----------
    models : dict
        Models.

    dataset : object
        Dataset.

    missing_rates : list
        Missing feature ratios.

    imputers : list
        Imputation strategies.

    task : str
        Problem type.

    random_state : int
        Seed.

    Returns
    -------
    pd.DataFrame
        Evaluation results.
    """
    results = []

    X_base = dataset.X.copy()

    dataset.X = X_base.copy()
    dataset.split(0.8, 0.0, 0.2)

    clean_metrics = {}

    for name, model in models.items():
        pipe = build_pipeline(clone(model))
        metrics = _run(pipe, dataset, task)
        clean_metrics[name] = metrics

        results.append({
            "model": name,
            "imputer": "none",
            "missing_rate": 0.0,
            "is_clean": True,
            **metrics
        })

    rng = np.random.default_rng(random_state)

    for rate in missing_rates:

        dataset.X = _corrupt(X_base, rate, rng)
        dataset.split(0.8, 0.0, 0.2)

        for imp in imputers:
            for name, model in models.items():

                pipe = build_pipeline(clone(model), imputer=imp)
                metrics = _run(pipe, dataset, task)

                base = {
                    "model": name,
                    "imputer": imp,
                    "missing_rate": rate,
                    "is_clean": False,
                    **metrics
                }

                for k, v in metrics.items():
                    if isinstance(v, (int, float)):

                        clean_v = clean_metrics[name].get(k, None)

                        if clean_v is not None:
                            direction = METRIC_DIRECTION.get(k, 1)

                            sens_abs = direction * (clean_v - v)

                            sens_rel = (
                                sens_abs / abs(clean_v)
                                if abs(clean_v) > 1e-12 else None
                            )

                            base[f"{k}_sens_abs"] = sens_abs
                            base[f"{k}_sens_rel"] = sens_rel

                results.append(base)

    return pd.DataFrame(results)


def convergence_analysis(model, dataset, max_iter):
    """
    Track training and validation loss over iterations.

    Parameters
    ----------
    model : estimator
        Model with training history.

    dataset : object
        Dataset.

    max_iter : int
        Maximum iterations.

    Returns
    -------
    dict
        Train and validation loss history.
    """
    dataset.split(0.6, 0.2, 0.2)
    model = clone(model)

    pipe = build_pipeline(
        model, None,
        predictor__max_iter=max_iter,
        predictor__store_history=True
    )

    scaler = StandardScaler().fit(dataset.X_train)
    X_val_scaled = scaler.transform(dataset.X_val)

    _run(
        pipe, dataset, task=None,
        model__predictor__X_val=X_val_scaled,
        model__predictor__y_val=dataset.y_val
    )

    return {
        "train_loss": getattr(
            model.named_steps["predictor"],
            "train_loss_history_", None
        ),
        "val_loss": getattr(
            model.named_steps["predictor"],
            "val_loss_history_", None
        )
    }


def plot_loss_curves(models, dataset, max_iters):
    """
    Plot convergence curves for multiple models.

    Parameters
    ----------
    models : dict
        Models.

    dataset : object
        Dataset.

    max_iters : list
        Iteration limits.

    Returns
    -------
    Figure
        Matplotlib figure.
    """
    n = len(models)
    ncols = int(np.ceil(np.sqrt(n)))
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(4 * ncols, 3 * nrows),
        constrained_layout=True
    )

    fig.suptitle("Convergence Curves (Train vs Validation Loss)")

    axes = axes.flatten()

    for i, (name, model) in enumerate(models.items()):

        hist = convergence_analysis(model, dataset, max_iters[i])

        axes[i].plot(hist["train_loss"], label="train")
        axes[i].plot(hist["val_loss"], label="val")

        axes[i].set_title(name)
        axes[i].set_xlabel("Epoch")
        axes[i].set_ylabel("Loss")
        axes[i].legend()

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    return fig


def plot_split_boxplots(df, metric):
    """
    Plot boxplots of metrics across splits.

    Parameters
    ----------
    df : DataFrame
        Results dataframe.

    metric : str
        Metric to visualize.

    Returns
    -------
    Figure
        Matplotlib figure.
    """
    models = df["model"].unique()
    n = len(models)

    ncols = int(np.ceil(np.sqrt(n)))
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(5 * ncols, 4 * nrows),
        constrained_layout=True
    )

    fig.suptitle(
        f"Sensitivity Analysis across Train/Test Splits "
        f"({metric.upper()})"
    )

    axes = np.array(axes).flatten()

    for i, name in enumerate(models):

        subset = df[df["model"] == name]

        sns.boxplot(
            data=subset,
            x="split",
            y=metric,
            ax=axes[i]
        )

        sns.stripplot(
            data=subset,
            x="split",
            y=metric,
            ax=axes[i],
            color="black",
            alpha=0.4,
            size=3
        )

        axes[i].set_title(name)
        axes[i].set_xlabel("Train/Test Split")
        axes[i].set_ylabel(metric)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    return fig
