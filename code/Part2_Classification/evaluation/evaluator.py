import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from statsmodels.stats.contingency_tables import mcnemar


class Evaluator:
    """
    Classification evaluation utility for sklearn-style models.

    This class provides:
    - Test set evaluation
    - Stratified K-Fold cross validation
    - Model comparison (CV and test)
    - Statistical significance testing (McNemar test)

    It supports pipelines, transformers, and estimators.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds used in Stratified K-Fold CV.

    random_state : int, default=42
        Random seed for reproducible splits.

    logger : object, default=None
        Optional logger for tracking evaluation steps.
    """

    def __init__(self, n_splits=5, random_state=42, logger=None,
                 log_model=False):
        """
        Initialize evaluator.

        Parameters
        ----------
        n_splits : int, default=5
            Number of CV folds.

        random_state : int, default=42
            Random seed for reproducibility.

        logger : object, default=None
            Logger for debugging and tracking.
        """
        self.n_splits = n_splits
        self.random_state = random_state
        self.logger = logger
        if log_model:
            self.model_fit_params = {"predictor__logger": self.logger}
        else:
            self.model_fit_params = {}

    # ====================== LOG ======================

    def _log(self, data):
        """
        Log information using the provided logger.

        Parameters
        ----------
        data : any
            Data to log.
        """
        if self.logger is not None:
            self.logger.info(data)

    def _log_model_info(self, model):
        """
        Log model structure and parameters recursively.

        Parameters
        ----------
        model : estimator
            sklearn-compatible model or pipeline.
        """
        def log_estimator(est, name=None):
            info = {
                "type": "model",
                "name": name,
                "class": est.__class__.__name__
            }

            if hasattr(est, "get_params"):
                params = est.get_params(deep=False)
                params = {
                    k: v for k, v in params.items()
                    if v is not None and not callable(v)
                }
                info["params"] = params

            self._log(info)

            if isinstance(est, Pipeline):
                for step_name, step in est.named_steps.items():
                    log_estimator(step, step_name)

            elif isinstance(est, ColumnTransformer):
                for name, trans, cols in est.transformers:
                    self._log({
                        "type": "column_transform",
                        "name": name,
                        "columns": cols
                    })

                    if trans not in ["drop", "passthrough"]:
                        log_estimator(trans, name)

        log_estimator(model)

    # ====================== METRICS ======================

    def evaluate(self, y_true, y_pred, y_proba=None, average="macro"):
        """
        Compute classification metrics.

        Parameters
        ----------
        y_true : array-like
            Ground truth labels.

        y_pred : array-like
            Predicted labels.

        y_proba : array-like, default=None
            Predicted probabilities for positive class.

        average : str, default="macro"
            Averaging method for multi-class metrics.

        Returns
        -------
        dict
            Accuracy, precision, recall, f1,
            per-class metrics, ROC AUC, confusion matrix.
        """
        acc = accuracy_score(y_true, y_pred)

        precision_avg = precision_score(
            y_true, y_pred, average=average, zero_division=0
        )
        recall_avg = recall_score(
            y_true, y_pred, average=average, zero_division=0
        )
        f1_avg = f1_score(
            y_true, y_pred, average=average, zero_division=0
        )

        precision_per_class = precision_score(
            y_true, y_pred, average=None, zero_division=0
        )
        recall_per_class = recall_score(
            y_true, y_pred, average=None, zero_division=0
        )
        f1_per_class = f1_score(
            y_true, y_pred, average=None, zero_division=0
        )

        cm = confusion_matrix(y_true, y_pred)

        roc_auc = None
        if y_proba is not None and len(np.unique(y_true)) == 2:
            try:
                roc_auc = roc_auc_score(y_true, y_proba)
            except:
                roc_auc = None

        metrics = {
            "Accuracy": acc,
            f"Precision ({average})": precision_avg,
            f"Recall ({average})": recall_avg,
            f"F1 ({average})": f1_avg,
            "Precision (per-class)": precision_per_class,
            "Recall (per-class)": recall_per_class,
            "F1 (per-class)": f1_per_class,
            "ROC AUC": roc_auc,
            "Confusion Matrix": cm
        }

        self._log({"type": "metrics", **metrics})

        return metrics

    # ====================== TEST ======================

    def evaluate_test(self, model, X_train, y_train, X_test, y_test,
                      average="macro"):
        """
        Evaluate model on held-out test set.

        Parameters
        ----------
        model : estimator
            Model to evaluate.

        X_train : array-like
            Training features.

        y_train : array-like
            Training labels.

        X_test : array-like
            Test features.

        y_test : array-like
            Test labels.

        average : str, default="macro"
            Averaging method for metrics.

        Returns
        -------
        dict
            Evaluation metrics on test set.
        """
        self._log({"type": "test_start"})

        self._log_model_info(model)

        self._log({
            "type": "split",
            "train_size": len(X_train),
            "test_size": len(X_test)
        })

        model = clone(model)

        model.fit(X_train, y_train, **self.model_fit_params)

        y_pred = model.predict(X_test)

        y_proba = None
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(X_test)
            if prob.shape[1] == 2:
                y_proba = prob[:, 1]

        return self.evaluate(y_test, y_pred, y_proba, average)

    # ====================== CROSS VALIDATION ======================

    def cross_validate(self, model, X, y, average="macro"):
        """
        Perform Stratified K-Fold cross validation.

        Parameters
        ----------
        model : estimator
            Model to evaluate.

        X : array-like
            Input features.

        y : array-like
            Target labels.

        average : str, default="macro"
            Averaging method for metrics.

        Returns
        -------
        dict
            Mean and std of CV metrics.
        """
        self._log({"type": "cv_start"})

        self._log_model_info(model)

        kf = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state
        )

        splits = list(kf.split(X, y))

        for i, (tr, va) in enumerate(splits):
            self._log({
                "type": "fold_split",
                "fold": i,
                "train_sample": tr[:5].tolist(),
                "val_sample": va[:5].tolist()
            })

        scores = cross_validate(
            clone(model),
            X,
            y,
            scoring=[
                "accuracy",
                f"precision_{average}",
                f"recall_{average}",
                f"f1_{average}"
            ],
            cv=splits,
            n_jobs=-1,
            params=self.model_fit_params
        )

        acc = scores["test_accuracy"]
        prec = scores[f"test_precision_{average}"]
        rec = scores[f"test_recall_{average}"]
        f1 = scores[f"test_f1_{average}"]

        for i in range(len(acc)):
            self._log({
                "type": "fold_metrics",
                "fold": i,
                "Accuracy": acc[i],
                f"Precision ({average})": prec[i],
                f"Recall ({average})": rec[i],
                f"F1 ({average})": f1[i]
            })

        return {
            "Accuracy": (np.mean(acc), np.std(acc)),
            f"Precision ({average})": (np.mean(prec), np.std(prec)),
            f"Recall ({average})": (np.mean(rec), np.std(rec)),
            f"F1 ({average})": (np.mean(f1), np.std(f1))
        }

    # ====================== COMPARE TEST ======================

    def compare_models_test(self, models, X_train, y_train, X_test,
                            y_test, average="macro"):
        """
        Compare multiple models on test set.

        Parameters
        ----------
        models : dict
            Dictionary of {name: estimator}.

        X_train : array-like
            Training features.

        y_train : array-like
            Training labels.

        X_test : array-like
            Test features.

        y_test : array-like
            Test labels.

        average : str, default="macro"
            Metric averaging method.

        Returns
        -------
        pandas.DataFrame
            Comparison of models on test metrics.
        """
        results = []

        for name, model in models.items():
            self._log({"type": "test_model_start", "model": name})

            metrics = self.evaluate_test(
                model, X_train, y_train, X_test, y_test, average
            )

            results.append({
                "Model": name,
                "Accuracy": metrics["Accuracy"],
                f"Precision ({average})":
                    metrics[f"Precision ({average})"],
                f"Recall ({average})":
                    metrics[f"Recall ({average})"],
                f"F1 ({average})":
                    metrics[f"F1 ({average})"],
                "ROC AUC": metrics.get("ROC AUC", None),
                "Precision_per_class":
                    metrics["Precision (per-class)"],
                "Recall_per_class":
                    metrics["Recall (per-class)"],
                "F1_per_class":
                    metrics["F1 (per-class)"],
                "Confusion Matrix":
                    metrics["Confusion Matrix"]
            })

        return pd.DataFrame(results)

    # ====================== COMPARE CV ======================

    def compare_models_cv(self, models, X, y, average="macro"):
        """
        Compare multiple models using cross validation.

        Parameters
        ----------
        models : dict
            Dictionary of models.

        X : array-like
            Input features.

        y : array-like
            Target labels.

        average : str, default="macro"
            Metric averaging method.

        Returns
        -------
        pandas.DataFrame
            CV results for all models.
        """
        results = []

        for name, model in models.items():
            self._log({"type": "cv_model_start", "model": name})

            scores = self.cross_validate(model, X, y, average)

            results.append({
                "Model": name,
                "Accuracy":
                    f"{scores['Accuracy'][0]:.4f} ± "
                    f"{scores['Accuracy'][1]:.4f}",
                f"Precision ({average})":
                    f"{scores[f'Precision ({average})'][0]:.4f} ± "
                    f"{scores[f'Precision ({average})'][1]:.4f}",
                f"Recall ({average})":
                    f"{scores[f'Recall ({average})'][0]:.4f} ± "
                    f"{scores[f'Recall ({average})'][1]:.4f}",
                f"F1 ({average})":
                    f"{scores[f'F1 ({average})'][0]:.4f} ± "
                    f"{scores[f'F1 ({average})'][1]:.4f}",
            })

        return pd.DataFrame(results)

    # ====================== MCNEMAR ======================

    def mcnemar_test(self, y_true, y_pred_a, y_pred_b):
        """
        Perform McNemar test for two classifiers.

        Parameters
        ----------
        y_true : array-like
            True labels.

        y_pred_a : array-like
            Predictions from model A.

        y_pred_b : array-like
            Predictions from model B.

        Returns
        -------
        dict
            McNemar statistic and p-value.
        """
        table = np.zeros((2, 2))

        for a, b, y in zip(y_pred_a, y_pred_b, y_true):
            table[int(a != y), int(b != y)] += 1

        result = mcnemar(table, exact=True)

        self._log({
            "type": "mcnemar",
            "table": table.tolist(),
            "statistic": result.statistic,
            "p_value": result.pvalue
        })

        return {
            "statistic": result.statistic,
            "p-value": result.pvalue
        }

    def compare_models_statistical(self, models, X_test, y_test):
        """
        Pairwise statistical comparison using McNemar test.

        Parameters
        ----------
        models : dict
            Dictionary of fitted models.

        X_test : array-like
            Test features.

        y_test : array-like
            True labels.

        Returns
        -------
        pandas.DataFrame
            Pairwise McNemar test results.
        """
        names = list(models.keys())
        results = []

        preds = {
            name: model.predict(X_test)
            for name, model in models.items()
        }

        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]

                self._log({
                    "type": "compare_pair",
                    "model_a": a,
                    "model_b": b
                })

                res = self.mcnemar_test(
                    y_test,
                    preds[a],
                    preds[b]
                )

                results.append({
                    "Model A": a,
                    "Model B": b,
                    "McNemar statistic": res["statistic"],
                    "p-value": res["p-value"]
                })

        return pd.DataFrame(results)
