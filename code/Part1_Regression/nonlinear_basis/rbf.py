import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

from code.Part1_Regression.dataset import CaliforniaHousingDataset as Dataset
from code.Part1_Regression.linear_regression.wls import WLS
from code.Part1_Regression.pipeline import get_pipeline

class RBF:
    def __init__(self, n_centers=10, gamma=0.1, seed=42):
        self.n_centers = n_centers
        self.gamma = gamma
        self.seed = seed

    def fit(self, X):
        kmeans = KMeans(n_clusters=self.n_centers,random_state=self.seed)
        kmeans.fit(X)
        self.centers = kmeans.cluster_centers_
        return self

    def transform(self, X):
        # (n_samples, n_centers)
        diff = X[:, None, :] - self.centers[None, :, :]
        dist_sq = np.sum(diff ** 2, axis=2)

        return np.exp(-self.gamma * dist_sq)
    
if __name__ == "__main__":
    import numpy as np
    from sklearn.preprocessing import StandardScaler

    from ..dataset import CaliforniaHousingDataset as Dataset
    from ..linear_regression.wls import WLS
    from ..pipeline import get_pipeline

    d = Dataset()
    d.split()

    X_train, y_train = d.X_train, d.y_train
    X_test, y_test = d.X_test, d.y_test
    # scale X
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # scale y
    y_scaler = StandardScaler()
    y_train = y_scaler.fit_transform(y_train.reshape(-1,1)).ravel()
    y_test = y_scaler.transform(y_test.reshape(-1,1)).ravel()

    # RBF
    rbf = RBF(n_centers=20, gamma=0.1)
    rbf.fit(X_train)

    X_train_rbf = rbf.transform(X_train)
    X_test_rbf = rbf.transform(X_test)

    # add bias
    X_train_rbf = np.hstack([np.ones((X_train_rbf.shape[0], 1)), X_train_rbf])
    X_test_rbf = np.hstack([np.ones((X_test_rbf.shape[0], 1)), X_test_rbf])

    model = get_pipeline(LinearRegression())
    model.fit(X_train_rbf, y_train)
    y_pred = model.predict(X_test_rbf)

    sse = np.sum(np.square(y_test - y_pred))
    mse = np.mean((y_test - y_pred) ** 2)
    print(f"MSE = {mse:.4f}")
    print(f"SSE = {sse:.4f}")
