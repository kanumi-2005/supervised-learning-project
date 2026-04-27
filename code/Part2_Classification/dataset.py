import numpy as np
from pandas.core.common import random_state
from pandas.errors import DatabaseError
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split


class CovtypeDataset:
    """
    Forest CoverType dataset loader and splitter.

    This class provides a simple wrapper around the
    sklearn Covtype dataset. It loads the dataset into
    memory and provides utilities to split it into
    train/validation/test subsets.

    Attributes
    ----------
    X : ndarray of shape (n_samples, n_features)
        Feature matrix.

    y : ndarray of shape (n_samples,)
        Target labels.

    df : pandas.DataFrame
        Full dataset including features and target.

    feature_names : list of str
        Names of input features.

    target_name : str
        Name of target variable.

    n_features : int
        Number of input features.

    n_samples : int
        Number of samples in dataset.

    classes : ndarray
        Unique class labels.

    X_train, y_train : ndarray
        Training set after split.

    X_val, y_val : ndarray or None
        Validation set after split.

    X_test, y_test : ndarray
        Test set after split.

    Examples
    --------
    >>> d = CovtypeDataset()
    >>> d.split()
    >>> d.train_size() > 0
    True
    """

    def __init__(self):
        """
        Load the Covtype dataset from sklearn.

        The dataset is loaded in-memory and converted
        into numpy arrays and pandas DataFrame.
        """
        dataset = fetch_covtype(as_frame=True)
        self.X = dataset.data.to_numpy()
        self.y = dataset.target.to_numpy()
        self.df = dataset.frame
        self.feature_names = dataset.feature_names
        self.target_name = dataset.target_names[0]
        self.n_features = self.X.shape[1]
        self.n_samples = self.X.shape[0]
        self.classes = np.unique(self.y)

    def size(self):
        """
        Return total number of samples in dataset.

        Returns
        -------
        int
            Number of samples.
        """
        return self.n_samples

    def train_size(self):
        """
        Return number of training samples.

        Returns
        -------
        int
            Number of training samples.
        """
        return self.X_train.shape[0]

    def val_size(self):
        """
        Return number of validation samples.

        Returns
        -------
        int
            Number of validation samples.
        """
        return self.X_val.shape[0]

    def test_size(self):
        """
        Return number of test samples.

        Returns
        -------
        int
            Number of test samples.
        """
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
        """
        Split dataset into train/validation/test sets.

        Parameters
        ----------
        train_size : float, default=0.6
            Proportion of training data.

        val_size : float, default=0.2
            Proportion of validation data.

        test_size : float, default=0.2
            Proportion of test data.

        random_state : int, default=42
            Random seed for reproducibility.

        shuffle : bool, default=True
            Whether to shuffle dataset before splitting.

        stratify : bool, default=True
            Whether to preserve class distribution.

        Returns
        -------
        None
            Splits data into internal attributes:
            X_train, X_val, X_test, y_train, y_val, y_test.
        """
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

        if val_size == 0 or X_temp.shape[0] == 0:
            self.X_train, self.y_train = X_train, y_train
            self.X_val, self.y_val = None, None
            self.X_test, self.y_test = X_temp, y_temp
            return

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
