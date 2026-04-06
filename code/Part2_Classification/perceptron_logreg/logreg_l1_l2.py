import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from code.Part2_Classification.dataset import CovtypeDataset

def compare_L1_L2():
    # ===== Load data =====
    d = CovtypeDataset()

    # Binary classification
    d.y = np.where(d.y == 1, 1, 0)
    d.split()

    X_train, y_train = d.X_train, d.y_train
    X_test, y_test = d.X_test, d.y_test

    # ===== Scale =====
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # ===== L1 Logistic Regression =====
    model_l1 = LogisticRegression(
        penalty='l1',
        solver='saga',
        max_iter=100
    )
    model_l1.fit(X_train, y_train)

    # ===== L2 Logistic Regression =====
    model_l2 = LogisticRegression(
        penalty='l2',
        solver='lbfgs',
        max_iter=100
    )
    model_l2.fit(X_train, y_train)

    # ===== Accuracy =====
    acc_l1 = accuracy_score(y_test, model_l1.predict(X_test))
    acc_l2 = accuracy_score(y_test, model_l2.predict(X_test))

    # ===== Sparsity =====
    w_l1 = model_l1.coef_
    w_l2 = model_l2.coef_

    sparsity_l1 = np.mean(w_l1 == 0)
    sparsity_l2 = np.mean(w_l2 == 0)

    print("=== RESULTS ===")
    print("L1 Accuracy:", acc_l1)
    print("L2 Accuracy:", acc_l2)
    print("L1 Sparsity:", sparsity_l1)
    print("L2 Sparsity:", sparsity_l2)

def main():
    compare_L1_L2()

if __name__ == "__main__":
    main()