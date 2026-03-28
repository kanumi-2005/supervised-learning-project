import numpy as np
from pandas.core.common import random_state
from pandas.errors import DatabaseError
from sklearn.datasets import fetch_covtype


class CovtypeDataset:

    def __init__(self):
        dataset = fetch_covtype(as_frame=True)
        self.X = dataset.data.to_numpy()
        self.y = dataset.target.to_numpy()
        self.df = dataset.frame
        self.feature_names = dataset.feature_names
        self.target_name = dataset.target_names[0]
        self.n_features = self.X.shape[1]
        self.n_samples = self.X.shape[0]

    def size(self):
        return self.n_samples

    def train_size(self):
        return self.X_train.shape[0]

    def val_size(self):
        return self.X_val.shape[0]

    def test_size(self):
        return self.X_test.shape[0]

    def split(
        self,
        train_size = 0.6,
        val_size = 0.2,
        test_size = 0.2,
        random_state = 42
    ):
        assert train_size + val_size + test_size == 1.0

        rng = np.random.default_rng(random_state)
        indices = rng.permutation(self.size())

        train_end = int(self.size() * train_size)
        val_end = train_end + int(self.size() * val_size)

        train_idx = indices[:train_end]
        val_idx = indices[train_end:val_end]
        test_idx = indices[val_end:]

        self.X_train = self.X[train_idx]
        self.X_val = self.X[val_idx]
        self.X_test = self.X[test_idx]
        self.y_train = self.y[train_idx]
        self.y_val = self.y[val_idx]
        self.y_test = self.y[test_idx]


if __name__ == "__main__":
    d = CovtypeDataset()

    print(d.df.describe())
