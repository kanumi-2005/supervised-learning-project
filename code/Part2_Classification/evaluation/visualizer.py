import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc, precision_recall_curve, average_precision_score
)
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import label_binarize


class Visualizer:

    def _ensure_ax(self, ax, title):
        if ax is None:
            fig, ax = plt.subplots(constrained_layout=True)
            ax.set_title(title)
            return fig, ax
        return None, ax

    def plot_loss_epochs(self, loss_list, val_loss_list=None, ax=None):
        fig, ax = self._ensure_ax(ax, "Loss vs Epochs")

        epochs = np.arange(1, len(loss_list) + 1)
        ax.plot(epochs, loss_list, label="Train Loss")
        if val_loss_list is not None:
            ax.plot(epochs, val_loss_list, label="Validation Loss")

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Cross-Entropy Loss")
        ax.legend()
        ax.grid(True)
        return fig, ax

    def plot_confusion_matrix(self, y_true, y_pred, labels, ax=None):
        fig, ax = self._ensure_ax(ax, "Confusion Matrix")

        cm = confusion_matrix(y_true, y_pred, labels=labels)
        disp = ConfusionMatrixDisplay(cm, display_labels=labels)
        disp.plot(ax=ax, cmap="Blues", colorbar=False)
        return fig, ax

    def plot_roc_curve(self, y_true, y_score, classes, ax=None):
        fig, ax = self._ensure_ax(ax, "ROC Curve (OvR)")

        y_bin = label_binarize(y_true, classes=classes)

        for i, c in enumerate(classes):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
            auc_score = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f"Class {c} (AUC={auc_score:.3f})")

        ax.plot([0, 1], [0, 1], "k--")
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.legend()
        ax.grid(True)
        return fig, ax

    def plot_precision_recall_curve(self, y_true, y_score, classes, ax=None):
        fig, ax = self._ensure_ax(ax, "PR Curve (OvR)")

        y_bin = label_binarize(y_true, classes=classes)

        for i, c in enumerate(classes):
            precision, recall, _ = precision_recall_curve(
                y_bin[:, i], y_score[:, i]
            )
            ap = average_precision_score(y_bin[:, i], y_score[:, i])
            ax.plot(recall, precision, label=f"Class {c} (AP={ap:.3f})")

        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.legend()
        ax.grid(True)
        return fig, ax

    def plot_calibration_curve(self, y_true, y_score, classes, ax=None,
                               n_bins=10):
        fig, ax = self._ensure_ax(ax, "Calibration Curve (OvR)")

        y_bin = label_binarize(y_true, classes=classes)

        for i, c in enumerate(classes):
            y_true_i = y_bin[:, i].astype(float)
            y_score_i = y_score[:, i].astype(float)

            if len(np.unique(y_true_i)) < 2:
                continue

            try:
                prob_true, prob_pred = calibration_curve(
                    y_true_i,
                    y_score_i,
                    n_bins=n_bins,
                    strategy="uniform"
                )
                ax.plot(prob_pred, prob_true, marker="o", label=f"Class {c}")
            except Exception:
                continue

        ax.plot([0, 1], [0, 1], "--", label="Perfect")
        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Fraction of Positives")
        ax.legend()
        ax.grid(True)
        return fig, ax

    def plot_decision_boundary(self, X, y, model, ax=None, resolution=200):
        fig, ax = self._ensure_ax(ax, "Decision Boundary")

        if X.shape[1] != 2:
            raise ValueError("X must have exactly 2 features")

        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, resolution),
            np.linspace(y_min, y_max, resolution)
        )

        grid = np.c_[xx.ravel(), yy.ravel()]
        Z = model.predict(grid)
        Z = Z.reshape(xx.shape)

        ax.contourf(xx, yy, Z, alpha=0.3)
        scatter = ax.scatter(X[:, 0], X[:, 1], c=y, s=10, edgecolor="k")

        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")

        return fig, ax


if __name__ == "__main__":
    from ..dataset import CovtypeDataset as Dataset
    from ..logistic_regression.softmax import SoftmaxClassifier

    d = Dataset()
    d.split()

    model = SoftmaxClassifier()
    model.fit(d.X_train, d.y_train)

    y_true = d.y_test
    y_pred = model.predict(d.X_test)
    y_score = model.predict_proba(d.X_test)

    visualizer = Visualizer()

    fig, axes = plt.subplots(2, 2, constrained_layout=True)

    visualizer.plot_confusion_matrix(
        y_true, y_pred, d.classes, ax=axes[0, 0]
    )
    visualizer.plot_roc_curve(
        y_true, y_score, d.classes, ax=axes[0, 1]
    )
    visualizer.plot_precision_recall_curve(
        y_true, y_score, d.classes, ax=axes[1, 0]
    )
    visualizer.plot_calibration_curve(
        y_true, y_score, d.classes, ax=axes[1, 1]
    )

    fig.suptitle("Model Evaluation - Covtype")
    plt.show()
