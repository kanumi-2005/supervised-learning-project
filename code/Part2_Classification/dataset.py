import numpy as np
from pandas.core.common import random_state
from pandas.errors import DatabaseError
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split


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
        train_size=0.6,
        val_size=0.2,
        test_size=0.2,
        random_state=42,
        shuffle=True,
        stratify=True,
    ):
        assert train_size + val_size + test_size == 1.0

        stratify_y = self.y if stratify else None

        X_train, X_temp, y_train, y_temp = train_test_split(
            self.X,
            self.y,
            train_size=train_size,
            random_state=random_state,
            shuffle=shuffle,
            stratify=stratify_y,
        )

        val_ratio = val_size / (val_size + test_size)
        stratify_temp = y_temp if stratify else None
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp,
            y_temp,
            train_size=val_ratio,
            random_state=random_state,
            shuffle=shuffle,
            stratify=stratify_temp,
        )

        self.X_train, self.y_train = X_train, y_train
        self.X_val, self.y_val = X_val, y_val
        self.X_test, self.y_test = X_test, y_test


if __name__ == "__main__":
    d = CovtypeDataset()

    print(d.df.describe())
