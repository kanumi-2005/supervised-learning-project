from sklearn.base import clone
import pandas as pd
import numpy as np


def benchmark_models(models, dataset, n_runs=5, random_state=42):
    """
    Benchmark models in terms of training and inference performance.

    This function evaluates multiple models over repeated random
    splits of a dataset and measures training/inference time and
    memory usage statistics from the model's predictor component.

    Parameters
    ----------
    models : dict
        Mapping from model name to sklearn-like estimator.

    dataset : object
        Dataset object containing X, y and split() method.

    n_runs : int, default=5
        Number of random runs for averaging results.

    random_state : int, default=42
        Seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Table containing mean and std of:
        - training time
        - training memory
        - inference time
        - inference memory

    Examples
    --------
    >>> benchmark_models(models, dataset, n_runs=3)
    """

    results = []

    X_base = dataset.X.copy()
    y_base = dataset.y.copy()

    rng = np.random.default_rng(random_state)
    seeds = rng.integers(0, 2**32 - 1, size=n_runs)

    for name, model in models.items():

        train_times = []
        train_memories = []
        infer_times = []
        infer_memories = []

        for seed in seeds:

            dataset.X = X_base.copy()
            dataset.y = y_base.copy()

            dataset.split(0.8, 0.0, 0.2, random_state=seed)

            m = clone(model)

            m.fit(dataset.X_train, dataset.y_train)
            m.predict(dataset.X_test)

            predictor = m.named_steps["predictor"]

            train_times.append(
                getattr(predictor, "training_time_", np.nan)
            )
            train_memories.append(
                getattr(predictor, "training_memory_", np.nan)
            )
            infer_times.append(
                getattr(predictor, "inference_time_", np.nan)
            )
            infer_memories.append(
                getattr(predictor, "inference_memory_", np.nan)
            )

        results.append({
            "model": name,
            "train_time_mean": np.mean(train_times),
            "train_time_std": np.std(train_times),

            "train_memory_mean": np.mean(train_memories),
            "train_memory_std": np.std(train_memories),

            "inference_time_mean": np.mean(infer_times),
            "inference_time_std": np.std(infer_times),

            "inference_memory_mean": np.mean(infer_memories),
            "inference_memory_std": np.std(infer_memories),
        })

    return pd.DataFrame(results)
