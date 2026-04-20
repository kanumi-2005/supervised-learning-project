import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.datasets import fetch_covtype
from sklearn.preprocessing import StandardScaler

from dataset import CovtypeDataset

def select_best_lambda(X, y, lambdas):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    best_lambda = None
    best_score = float("inf")

    for lam in lambdas:
        fold_scores = []
        C = 1.0 / lam

        model = LogisticRegression(
            C=C,
            penalty='l2',
            solver='lbfgs',
            max_iter=1000
        )

        for train_idx, val_idx in skf.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # ===== Scale =====
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)

            model.fit(X_train, y_train)
            y_prob = model.predict_proba(X_val)

            loss = log_loss(y_val, y_prob)
            fold_scores.append(loss)

        avg_loss = np.mean(fold_scores)
        print(f"Lambda = {lam}, Log Loss = {avg_loss:.4f}")

        if avg_loss < best_score:
            best_score = avg_loss
            best_lambda = lam

    return best_lambda, best_score

from sklearn.datasets import fetch_covtype
import numpy as np


def main():
    # ===== Load Covtype dataset =====
    d = CovtypeDataset()
    d.y = np.where(d.y == 1, 1, 0)
    X = d.X
    y = d.y
    print("Dataset shape:", X.shape)
    print("Class distribution:", np.bincount(y))

    # ===== Lambda=====
    lambdas = [0.001, 0.01, 0.1, 1, 10]

    # ===== Function choose best lambda =====
    best_lambda, best_score = select_best_lambda(X, y, lambdas)

    print("\n=== RESULT ===")
    print("Best lambda:", best_lambda)
    print("Best log loss:", best_score)


if __name__ == "__main__":
    main()