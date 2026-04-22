import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score
)
from sklearn.model_selection import KFold, cross_validate
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
        precision = precision_score(y_true, y_pred, average=average, zero_division=0)
        recall = recall_score(y_true, y_pred, average=average, zero_division=0)
        f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
        cm = confusion_matrix(y_true, y_pred)

        roc_auc = None
        if y_proba is not None and len(np.unique(y_true)) == 2:
            try:
                roc_auc = roc_auc_score(y_true, y_proba)
            except:
                roc_auc = None

        metrics = {
            "Accuracy": acc,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "ROC AUC": roc_auc,
            "Confusion Matrix": cm
        }

        self._log({
            "type": "metrics",
            **{k: v for k, v in metrics.items() if k != "Confusion Matrix"}
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

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        y_proba = None
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(X_test)
            if prob.shape[1] == 2:
                y_proba = prob[:, 1]

        return self.evaluate(y_test, y_pred, y_proba, average)

    # ====================== CROSS VALIDATION ======================

    def cross_validate(self, model, X, y):
        self._log({"type": "cv_start"})

        self._log_model_info(model)

        kf = KFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state
        )

        splits = list(kf.split(X))

        for i, (train_idx, val_idx) in enumerate(splits):
            self._log({
                "type": "fold_split",
                "fold": i,
                "train_sample": train_idx[:5].tolist(),
                "val_sample": val_idx[:5].tolist()
            })

        scores = cross_validate(
            model,
            X,
            y,
            scoring=[
                "accuracy",
                "precision_macro",
                "recall_macro",
                "f1_macro"
            ],
            cv=splits,
            n_jobs=-1
        )

        acc = scores["test_accuracy"]
        prec = scores["test_precision_macro"]
        rec = scores["test_recall_macro"]
        f1 = scores["test_f1_macro"]

        for i in range(len(acc)):
            self._log({
                "type": "fold_metrics",
                "fold": i,
                "Accuracy": acc[i],
                "Precision": prec[i],
                "Recall": rec[i],
                "F1": f1[i]
            })

        return {
            "Accuracy": (np.mean(acc), np.std(acc)),
            "Precision": (np.mean(prec), np.std(prec)),
            "Recall": (np.mean(rec), np.std(rec)),
            "F1": (np.mean(f1), np.std(f1))
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

            results.append({
                "Model": name,
                **metrics
            })

        return pd.DataFrame(results)

    # ====================== COMPARE CV ======================

    def compare_models_cv(self, models, X, y):
        results = []

        for name, model in models.items():
            self._log({
                "type": "cv_model_start",
                "model": name
            })

            scores = self.cross_validate(model, X, y)

            results.append({
                "Model": name,
                "Accuracy": f"{scores['Accuracy'][0]:.4f} ± {scores['Accuracy'][1]:.4f}",
                "Precision": f"{scores['Precision'][0]:.4f} ± {scores['Precision'][1]:.4f}",
                "Recall": f"{scores['Recall'][0]:.4f} ± {scores['Recall'][1]:.4f}",
                "F1": f"{scores['F1'][0]:.4f} ± {scores['F1'][1]:.4f}"
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
