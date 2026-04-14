from sklearn.base import BaseEstimator
import time
import tracemalloc
from contextlib import contextmanager


class BaseModel(BaseEstimator):
    def fit(self, X, y, logger=None, **kwargs):
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
        raise NotImplementedError

    def _predict(self, X):
        raise NotImplementedError

    def _log(self, data):
        if self._logger is not None:
            self._logger.info(data)

    @contextmanager
    def _track_resources(self):
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
