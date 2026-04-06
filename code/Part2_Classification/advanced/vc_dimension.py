import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.datasets import fetch_covtype
from sklearn.preprocessing import StandardScaler
from code.Part2_Classification.dataset import CovtypeDataset

def run_vc_srm():

    # ====== Load dataset từ class của bạn ======
    d = CovtypeDataset()

    # ====== Convert binary ======
    d.y = np.where(d.y == 1, 1, 0)

    # ====== Split ======
    d.split()

    X_train, y_train = d.X_train, d.y_train
    X_val, y_val = d.X_val, d.y_val
    X_test, y_test = d.X_test, d.y_test

    # ====== VC dimension ======
    D = d.n_features
    vc_dim = D + 1

    print("===== VC DIMENSION =====")
    print(f"D = {D}")
    print(f"VC = {vc_dim}")

    # ====== SRM ======
    C_values = [0.001, 0.01, 0.1, 1, 10, 100]

    train_acc = []
    val_acc = []
    test_acc = []

    for C in C_values:
        model = LogisticRegression(C=C, max_iter=200)

        model.fit(X_train, y_train)

        train_acc.append(accuracy_score(y_train, model.predict(X_train)))
        val_acc.append(accuracy_score(y_val, model.predict(X_val)))
        test_acc.append(accuracy_score(y_test, model.predict(X_test)))

        print(f"\nC = {C}")
        print("Train:", train_acc[-1])
        print("Val  :", val_acc[-1])
        print("Test :", test_acc[-1])

    # ====== Choose model from SRM ======
    best_idx = np.argmax(val_acc)
    best_C = C_values[best_idx]

    print("\n===== SRM RESULT =====")
    print(f"Best C = {best_C}")
    print(f"Best Val Acc = {val_acc[best_idx]}")
    print(f"Test Acc (best model) = {test_acc[best_idx]}")

    # ====== Plot ======
    plt.plot(C_values, train_acc, marker='o', label="Train")
    plt.plot(C_values, val_acc, marker='o', label="Validation")
    plt.plot(C_values, test_acc, marker='o', label="Test")

    plt.xscale('log')
    plt.xlabel("Model Complexity (C ~ VC effective)")
    plt.ylabel("Accuracy")
    plt.title("SRM on Covtype Dataset")
    plt.legend()
    plt.show()

def main():
    run_vc_srm()
if __name__ == "__main__":
    main()