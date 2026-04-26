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
    def __init__(self, n_splits=5, random_state=42, logger=None):
        self.n_splits = n_splits
        self.random_state = random_state
        self.logger = logger

    # ====================== LOG ======================

    def _log(self, data):
        if self.logger is not None:
            self.logger.info(data)

    def _log_model_info(self, model):
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
        acc = accuracy_score(y_true, y_pred)

        # ===== averaged metrics =====
        precision_avg = precision_score(y_true, y_pred, average=average,
                                        zero_division=0)
        recall_avg = recall_score(y_true, y_pred, average=average,
                                  zero_division=0)
        f1_avg = f1_score(y_true, y_pred, average=average, zero_division=0)

        # ===== per-class metrics =====
        precision_per_class = precision_score(y_true, y_pred, average=None,
                                              zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None,
                                        zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)

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

        self._log({
            "type": "metrics",
            **{k: v for k, v in metrics.items()}
        })

        return metrics

    # ====================== TEST ======================

    def evaluate_test(self, model, X_train, y_train, X_test, y_test,
                      average="macro"):
        self._log({"type": "test_start"})

        self._log_model_info(model)

        self._log({
            "type": "split",
            "train_size": len(X_train),
            "test_size": len(X_test)
        })

        model = clone(model)

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        y_proba = None
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(X_test)
            if prob.shape[1] == 2:
                y_proba = prob[:, 1]

        return self.evaluate(y_test, y_pred, y_proba, average)

    # ====================== CROSS VALIDATION ======================

    def cross_validate(self, model, X, y, average="macro"):
        self._log({"type": "cv_start"})

        self._log_model_info(model)

        kf = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state
        )

        splits = list(kf.split(X, y))

        for i, (train_idx, val_idx) in enumerate(splits):
            self._log({
                "type": "fold_split",
                "fold": i,
                "train_sample": train_idx[:5].tolist(),
                "val_sample": val_idx[:5].tolist()
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
            n_jobs=-1
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

    def compare_models_test(self, models, X_train, y_train, X_test, y_test,
                            average="macro"):
        results = []

        for name, model in models.items():
            self._log({
                "type": "test_model_start",
                "model": name
            })

            metrics = self.evaluate_test(
                model,
                X_train,
                y_train,
                X_test,
                y_test,
                average
            )

            row = {
                "Model": name,
                "Accuracy": metrics["Accuracy"],

                f"Precision ({average})": metrics[f"Precision ({average})"],
                f"Recall ({average})": metrics[f"Recall ({average})"],
                f"F1 ({average})": metrics[f"F1 ({average})"],

                "ROC AUC": metrics.get("ROC AUC", None),

                # ===== FULL PER-CLASS (FLAT) =====
                "Precision_per_class": metrics["Precision (per-class)"],
                "Recall_per_class": metrics["Recall (per-class)"],
                "F1_per_class": metrics["F1 (per-class)"],

                # ===== CONFUSION MATRIX =====
                "Confusion Matrix": metrics["Confusion Matrix"]
            }

            results.append(row)

        return pd.DataFrame(results)

    # ====================== COMPARE CV ======================

    def compare_models_cv(self, models, X, y, average="macro"):
        results = []

        for name, model in models.items():
            self._log({
                "type": "cv_model_start",
                "model": name
            })

            scores = self.cross_validate(model, X, y, average=average)

            results.append({
                "Model": name,

                "Accuracy":
                    f"{scores['Accuracy'][0]:.4f} ± " \
                    f"{scores['Accuracy'][1]:.4f}",

                f"Precision ({average})":
                    f"{scores[f'Precision ({average})'][0]:.4f} ± " \
                    f"{scores[f'Precision ({average})'][1]:.4f}",

                f"Recall ({average})":
                    f"{scores[f'Recall ({average})'][0]:.4f} ± " \
                    f"{scores[f'Recall ({average})'][1]:.4f}",

                f"F1 ({average})":
                    f"{scores[f'F1 ({average})'][0]:.4f} ± " \
                    f"{scores[f'F1 ({average})'][1]:.4f}",
            })

        return pd.DataFrame(results)

    # ====================== MCNEMAR ======================

    def mcnemar_test(self, y_true, y_pred_a, y_pred_b):
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

        return {"statistic": result.statistic, "p-value": result.pvalue}

    def compare_models_statistical(self, models, X_test, y_test):
        names = list(models.keys())
        results = []

        # ⚠️ assume models already fitted
        preds = {name: model.predict(X_test) for name, model in models.items()}

        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                name_a, name_b = names[i], names[j]

                self._log({
                    "type": "compare_pair",
                    "model_a": name_a,
                    "model_b": name_b
                })

                res = self.mcnemar_test(
                    y_test,
                    preds[name_a],
                    preds[name_b]
                )

                results.append({
                    "Model A": name_a,
                    "Model B": name_b,
                    "McNemar statistic": res["statistic"],
                    "p-value": res["p-value"]
                })

        return pd.DataFrame(results)


if __name__ == "__main__":
    from ..dataset import CovtypeDataset as Dataset
    from ..logistic_regression.softmax import SoftmaxClassifier
    from ..lda.lda import LDA
    from ..lda.qda import QDA

    d = Dataset()
    d.split()

    models = {
        "softmax": SoftmaxClassifier(),
        "lda": LDA(),
    #    "qda": QDA()
    }

    evaluator = Evaluator()

    results = evaluator.compare_models_test(
        models,
        d.X_train,
        d.y_train,
        d.X_test,
        d.y_test
    )

    print(results)

    results = evaluator.compare_models_cv(
        models,
        d.X_train,
        d.y_train
    )

    print(results)

    results = evaluator.compare_models_statistical(
        models,
        d.X_test,
        d.y_test
    )

    print(results)
