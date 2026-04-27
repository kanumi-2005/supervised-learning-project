from sklearn.base import BaseEstimator
import time
import tracemalloc
from contextlib import contextmanager


class BaseModel(BaseEstimator):
    """
    Base estimator class with training/inference tracking.

    This class extends sklearn BaseEstimator and provides a unified
    interface for models with built-in logging and resource tracking
    (time and memory usage) during training and prediction.

    Subclasses must implement `_fit` and `_predict`.

    Parameters
    ----------
    None

    Attributes
    ----------
    training_time_ : float
        Time spent during training (seconds).

    training_memory_ : int
        Peak memory usage during training (bytes).

    inference_time_ : float
        Time spent during inference (seconds).

    inference_memory_ : int
        Peak memory usage during inference (bytes).

    _logger : object
        Optional logger for structured logging.

    Examples
    --------
    >>> class Model(BaseModel):
    ...     def _fit(self, X, y): return None
    ...     def _predict(self, X): return X
    """

    def fit(self, X, y, logger=None, **kwargs):
        """
        Fit model with resource tracking.

        Parameters
        ----------
        X : array-like
            Training features.

        y : array-like
            Target values.

        logger : object, default=None
            Logger for tracking events.

        **kwargs :
            Additional parameters for `_fit`.

        Returns
        -------
        self : object
            Fitted model.
        """
        self._logger = logger

        self._log({
            "type": "config",
            "params": self.get_params()
        })

        with self._track_resources() as train_metrics:
            result = self._fit(X, y, **kwargs)

            if result is not None:
                for state in result:
                    self._log({
                        "type": "step",
                        **state
                    })

        self.training_time_ = train_metrics["time"]
        self.training_memory_ = train_metrics["memory"]

        self._log({
            "type": "train",
            "time": self.training_time_,
            "memory": self.training_memory_
        })

        return self

    def predict(self, X, logger=None):
        """
        Predict with resource tracking.

        Parameters
        ----------
        X : array-like
            Input features.

        logger : object, default=None
            Logger instance.

        Returns
        -------
        array-like
            Predictions.
        """
        self._logger = logger

        with self._track_resources() as infer_metrics:
            y_pred = self._predict(X)

        self.inference_time_ = infer_metrics["time"]
        self.inference_memory_ = infer_metrics["memory"]

        self._log({
            "type": "inference",
            "time": self.inference_time_,
            "memory": self.inference_memory_
        })

        return y_pred

    def _fit(self, X, y, **kwargs):
        """
        Internal training method.

        Must be implemented by subclasses.

        Parameters
        ----------
        X : array-like
        y : array-like
        """
        raise NotImplementedError

    def _predict(self, X):
        """
        Internal prediction method.

        Must be implemented by subclasses.

        Parameters
        ----------
        X : array-like
        """
        raise NotImplementedError

    def _log(self, data):
        """
        Log structured information if logger exists.

        Parameters
        ----------
        data : dict
            Log message.
        """
        if self._logger is not None:
            self._logger.info(data)

    @contextmanager
    def _track_resources(self):
        """
        Context manager for tracking time and memory usage.

        Measures execution time and peak memory usage using tracemalloc.

        Yields
        ------
        dict
            Dictionary containing 'time' and 'memory'.
        """
        tracemalloc.start()
        start_time = time.perf_counter()

        metrics = {}

        try:
            yield metrics
        finally:
            end_time = time.perf_counter()
            _, peak_memory = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            metrics["time"] = end_time - start_time
            metrics["memory"] = peak_memory
