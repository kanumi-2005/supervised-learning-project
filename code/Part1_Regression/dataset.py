import numpy as np
from sklearn.datasets import fetch_california_housing


class CaliforniaHousingDataset:
    """
    California Housing dataset wrapper with train/val/test splitting.

    This class loads the California Housing dataset from sklearn and
    provides utilities for splitting it into train, validation, and
    test sets using random permutation of indices.

    Parameters
    ----------
    None

    Attributes
    ----------
    X : ndarray of shape (n_samples, n_features)
        Feature matrix.

    y : ndarray of shape (n_samples,)
        Target values.

    df : pandas.DataFrame
        Full dataset as a dataframe including features and target.

    feature_names : list of str
        Names of input features.

    target_name : str
        Name of the target variable.

    n_features : int
        Number of features.

    n_samples : int
        Number of samples.

    X_train : ndarray
        Training features after split.

    X_val : ndarray
        Validation features after split.

    X_test : ndarray
        Test features after split.

    y_train : ndarray
        Training targets after split.

    y_val : ndarray
        Validation targets after split.

    y_test : ndarray
        Test targets after split.

    Examples
    --------
    >>> ds = CaliforniaHousingDataset()
    >>> ds.split(train_size=0.6, val_size=0.2, test_size=0.2)
    >>> ds.train_size()
    12384
    >>> ds.test_size()
    4128
    """

    def __init__(self):
        """
        Load California Housing dataset.
        """
        dataset = fetch_california_housing(as_frame=True)
        self.X = dataset.data.to_numpy()
        self.y = dataset.target.to_numpy()
        self.df = dataset.frame
        self.feature_names = dataset.feature_names
        self.target_name = dataset.target_names[0]
        self.n_features = self.X.shape[1]
        self.n_samples = self.X.shape[0]

    def size(self) -> int:
        """
        Return total number of samples.

        Returns
        -------
        int
            Number of samples in dataset.
        """
        return len(self.X)

    def train_size(self) -> int:
        """
        Return number of training samples.

        Returns
        -------
        int
            Size of training set.
        """
        return len(self.X_train)

    def val_size(self) -> int:
        """
        Return number of validation samples.

        Returns
        -------
        int
            Size of validation set.
        """
        return len(self.X_val)

    def test_size(self) -> int:
        """
        Return number of test samples.

        Returns
        -------
        int
            Size of test set.
        """
        return len(self.X_test)

    def split(
        self,
        train_size: float = 0.6,
        val_size: float = 0.2,
        test_size: float = 0.2,
        random_state: int = 42,
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

        Returns
        -------
        None
            Sets X_train, X_val, X_test, y_train,
            y_val, y_test as attributes.
        """
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
