import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, make_scorer
)
from sklearn.model_selection import KFold, cross_validate
from statsmodels.stats.contingency_tables import mcnemar


class Evaluator:
    def __init__(self, n_splits=5, random_state=42):
        self.n_splits = n_splits
        self.random_state = random_state

    def evaluate(self, y_true, y_pred, average="macro"):
        acc = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average=average)
        recall = recall_score(y_true, y_pred, average=average)
        f1 = f1_score(y_true, y_pred, average=average)
        cm = confusion_matrix(y_true, y_pred)

        roc_auc = None
        if len(np.unique(y_true)) == 2:
            roc_auc = roc_auc_score(y_true, y_pred)

        return {
            "Accuracy": acc,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "Confusion Matrix": cm,
            "ROC AUC": roc_auc
        }

    def evaluate_test(self, model, X_train, y_train, X_test, y_test,
                      average="macro"):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return self.evaluate(y_test, y_pred, average=average)

    def cross_validate(self, model, X, y):
        scoring = [
            "accuracy",
            "precision_macro",
            "recall_macro",
            "f1_macro"
        ]
        scores = cross_validate(
            model,
            X,
            y,
            scoring=scoring,
            cv=KFold(n_splits=self.n_splits, shuffle=True,
                     random_state=self.random_state),
            n_jobs=-1
        )

        return {
            "Accuracy": (
                np.mean(scores["test_accuracy"]),
                np.std(scores["test_accuracy"])
            ),
            "Precision": (
                np.mean(scores["test_precision_macro"]),
                np.std(scores["test_precision_macro"])
                ),
            "Recall": (
                np.mean(scores["test_recall_macro"]),
                np.std(scores["test_recall_macro"])
                ),
            "F1": (
                np.mean(scores["test_f1_macro"]),
                np.std(scores["test_f1_macro"])
            )
        }

    def compare_models_test(self, models, X_train, y_train, X_test, y_test,
                            average="macro"):
        results = []
        for name, model in models.items():
            metrics = self.evaluate_test(model, X_train, y_train, X_test,
                                         y_test, average=average)
            results.append({
                "Model": name,
                "Accuracy": metrics["Accuracy"],
                "Precision": metrics["Precision"],
                "Recall": metrics["Recall"],
                "F1": metrics["F1"],
                "ROC AUC": metrics["ROC AUC"],
                "Confusion Matrix": metrics["Confusion Matrix"]
            })
        return pd.DataFrame(results)

    def compare_models_cv(self, models, X, y):
        results = []
        for name, model in models.items():
            scores = self.cross_validate(model, X, y)
            results.append({
                "Model": name,
                "Accuracy": f"{scores['Accuracy'][0]:.4f} ± " \
                    f"{scores['Accuracy'][1]:.4f}",
                "Precision": f"{scores['Precision'][0]:.4f} ± " \
                    f"{scores['Precision'][1]:.4f}",
                "Recall": f"{scores['Recall'][0]:.4f} ± " \
                    f"{scores['Recall'][1]:.4f}",
                "F1": f"{scores['F1'][0]:.4f} ± {scores['F1'][1]:.4f}"
            })
        return pd.DataFrame(results)

    def mcnemar_test(self, model_a, model_b, X_test, y_test):
        y_pred_a = model_a.predict(X_test)
        y_pred_b = model_b.predict(X_test)

        table = np.zeros((2,2))
        for a, b, y in zip(y_pred_a, y_pred_b, y_test):
            table[int(a != y), int(b != y)] += 1

        result = mcnemar(table, exact=True)
        return {"statistic": result.statistic, "p-value": result.pvalue}

    def compare_models_statistical(self, models, X_test, y_test):
        names = list(models.keys())
        results = []

        for i in range(len(names)):
            for j in range(i+1, len(names)):
                name_a, name_b = names[i], names[j]
                model_a = models[name_a]
                model_b = models[name_b]

                model_a.fit(X_test, y_test)
                model_b.fit(X_test, y_test)

                mcnemar_res = self.mcnemar_test(model_a, model_b, X_test,
                                                y_test)

                results.append({
                    "Model A": name_a,
                    "Model B": name_b,
                    "McNemar statistic": mcnemar_res["statistic"],
                    "p-value": mcnemar_res["p-value"]
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
