import numpy as np
from sklearn.datasets import fetch_california_housing


class CaliforniaHousingDataset:
    """
    Utility class for loading and splitting the California Housing dataset.

    Attributes
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target vector of shape (n_samples,).
    df : pandas.DataFrame
        Full dataset as a pandas DataFrame.
    """

    def __init__(self):
        """
        Load the California Housing dataset into memory.
        """
        dataset = fetch_california_housing(as_frame=True)
        self.X = dataset.data.to_numpy()
        self.y = dataset.target.to_numpy()
        self.df = dataset.frame

    def size(self) -> int:
        """
        Return the number of samples in the dataset.

        Returns
        -------
        int
            Number of samples.
        """
        return len(self.X)

    def split(
        self,
        train_size: float = 0.6,
        val_size: float = 0.2,
        test_size: float = 0.2,
        random_state: int = 42,
    ):
        """
        Split the dataset into train, validation, and test sets.

        Parameters
        ----------
        train_size : float, default=0.6
            Proportion of data used for training.
        val_size : float, default=0.2
            Proportion of data used for validation.
        test_size : float, default=0.2
            Proportion of data used for testing.
        random_state : int, default=42
            Random seed for reproducibility.

        Raises
        ------
        AssertionError
            If train_size + val_size + test_size != 1.0

        Notes
        -----
        This method shuffles the dataset before splitting.

        After calling this method, the following attributes are available:
        - X_train, X_val, X_test
        - y_train, y_val, y_test
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
