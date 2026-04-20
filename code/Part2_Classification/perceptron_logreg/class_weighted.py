import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from dataset import CovtypeDataset
from sklearn.preprocessing import StandardScaler

class WeightedSoftmaxClassier:
    def __init__(self,lr = 0.01,epochs = 1000):
        self.lr = lr
        self.epochs = epochs

    def _softmax(self,z):
        exp_z = np.exp(z - np.max(z,axis=1,keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    def _one_hot(self,y,K):
        one_hot = np.zeros((len(y),K))
        one_hot[np.arange(len(y)), y] = 1
        return one_hot
    def _compute_class_weights(self,y,K):
        N = len(y)
        class_counts = np.bincount(y, minlength=K)
        weights = N / (class_counts + 1e-8)   # tránh chia 0
        return weights 
    def fit(self,X,y):
        N,D = X.shape
        K = len(np.unique(y))

        self.W = np.zeros((D,K))
        self.b = np.zeros(K)

        y_onehot = self._one_hot(y, K)
        class_weights = self._compute_class_weights(y, K)

        for epoch in range(self.epochs):
            # Forward
            logits = X @ self.W + self.b
            y_pred = self._softmax(logits)

            # ===== Weighted Loss =====
            loss = -np.sum(
                y_onehot * np.log(y_pred + 1e-8) * class_weights
            ) / N

            # ===== Gradient =====
            error = (y_pred - y_onehot) * class_weights

            dW = X.T @ error / N
            db = np.sum(error, axis=0) / N

            # Update
            self.W -= self.lr * dW
            self.b -= self.lr * db

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

    def predict(self, X):
        logits = X @ self.W + self.b
        y_pred = self._softmax(logits)
        return np.argmax(y_pred, axis=1)
    
if __name__ == "__main__":

    # ===== Load dataset =====
    d = CovtypeDataset()
    X = d.X
    y = d.y - 1   

    # ===== Split =====
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    # ===== Normalize (RẤT QUAN TRỌNG) =====
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # ===== Model no weight =====
    print("========== WITHOUT CLASS WEIGHT ==========")
    model_no_weight = WeightedSoftmaxClassier(lr=0.05, epochs=500)

    model_no_weight._compute_class_weights = lambda y, K: np.ones(K)

    model_no_weight.fit(X_train, y_train)
    y_pred_no = model_no_weight.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred_no))
    print(classification_report(y_test, y_pred_no, digits=4))

    # ===== Model contain weight =====
    print("\n========== WITH CLASS WEIGHT ==========")
    model_weight = WeightedSoftmaxClassier(lr=0.05, epochs=500)

    model_weight.fit(X_train, y_train)
    y_pred_w = model_weight.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred_w))
    print(classification_report(y_test, y_pred_w, digits=4))

    # ===== In class weights =====
    print("\n========== CLASS WEIGHTS ==========")
    K = len(np.unique(y_train))
    weights = model_weight._compute_class_weights(y_train, K)
    for i, w in enumerate(weights):
        print(f"Class {i}: weight = {w:.4f}")