from .basemodel import BaseModel
import numpy as np


class BaseGDModel(BaseModel):
    """
    Base class for gradient descent optimization models.

    This class provides a generic training loop for models optimized using
    gradient descent variants, including stochastic and mini-batch methods.
    It supports logging, resource tracking, and optional history storage for
    training and validation loss.

    The optimization procedure is delegated to subclass implementations via
    `_init_params`, `_grad`, `_update_params`, and `_loss`.

    Parameters
    ----------
    lr : float
        Learning rate used for parameter updates.

    max_iter : int
        Maximum number of training iterations (epochs).

    store_history : bool
        If True, stores training and validation loss history.

    batch_size : int or None, default=None
        Size of mini-batches used for gradient computation.
        If None, full-batch gradient descent is used.

    random_state : int, default=42
        Seed used for reproducible shuffling of training data.

    Attributes
    ----------
    lr : float
        Learning rate.

    max_iter : int
        Maximum number of iterations.

    store_history : bool
        Whether loss history is stored.

    batch_size : int or None
        Batch size for gradient computation.

    random_state : int
        Random seed.

    rng : np.random.Generator
        Random number generator used for shuffling.

    n_samples : int
        Number of training samples.

    train_loss_history_ : list of float
        Training loss values per iteration (if enabled).

    val_loss_history_ : list of float
        Validation loss values per iteration (if enabled).

    n_iter_ : int
        Number of iterations performed during training.

    Notes
    -----
    The training loop follows:
    1. Shuffle dataset each iteration.
    2. Split into mini-batches.
    3. Compute gradient per batch.
    4. Update parameters.
    5. Compute and log losses.

    Examples
    --------
    >>> model = MyGDModel(lr=0.01, max_iter=100,
    ...                   store_history=True)
    >>> model.fit(X_train, y_train)
    >>> y_pred = model.predict(X_test)
    """

    def __init__(
            self,
            lr,
            max_iter,
            store_history,
            batch_size=None,
            random_state=42
        ):
        """
        Initialize gradient descent base model.

        Parameters
        ----------
        lr : float
            Learning rate for updates.

        max_iter : int
            Maximum number of iterations.

        store_history : bool
            Whether to store loss history.

        batch_size : int or None, default=None
            Mini-batch size. If None, uses full batch.

        random_state : int, default=42
            Random seed for reproducibility.
        """
        super().__init__()
        self.lr = lr
        self.max_iter = max_iter
        self.store_history = store_history
        self.batch_size = batch_size
        self.random_state = random_state
        self.rng = np.random.default_rng(self.random_state)

    def _fit(self, X, y, **kwargs):
        """
        Fit model using gradient descent optimization loop.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,)
            Target values.

        **kwargs : dict
            Optional validation data:
            X_val : array-like, optional
                Validation features.
            y_val : array-like, optional
                Validation targets.

        Yields
        ------
        dict
            Dictionary containing:
            - iteration index
            - training loss
            - validation loss (if available)
            - optional extra logs from `_extra_logs`

        Returns
        -------
        None
        """
        X_val = kwargs.get("X_val", None)
        y_val = kwargs.get("y_val", None)

        self._init_params(X, y)
        self.n_samples = X.shape[0]

        if self.store_history:
            self.train_loss_history_ = []
            self.val_loss_history_ = []

        batch_size = self.batch_size or self.n_samples

        for it in range(self.max_iter):
            indices = self.rng.permutation(self.n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for start in range(0, self.n_samples, batch_size):
                end = start + batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                grad = self._grad(X_batch, y_batch)
                self._update_params(grad, iteration=it)

            train_loss = self._loss(X, y)

            val_loss = None
            if X_val is not None and y_val is not None:
                val_loss = self._loss(X_val, y_val)

            if self.store_history:
                self.train_loss_history_.append(train_loss)
                if val_loss is not None:
                    self.val_loss_history_.append(val_loss)

            base_log = {
                "iter": it,
                "train_loss": train_loss,
                "val_loss": val_loss
            }

            extra_log = self._extra_logs(
                X=X,
                y=y,
                grad=None,
                iter=it
            )

            if extra_log:
                base_log.update(extra_log)

            yield base_log

        self.n_iter_ = self.max_iter

    def _init_params(self, X, y):
        """
        Initialize model parameters.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        None
        """
        raise NotImplementedError

    def _loss(self, X, y):
        """
        Compute loss function value.

        Parameters
        ----------
        X : array-like
            Input data.

        y : array-like
            Target values.

        Returns
        -------
        float
            Computed loss value.
        """
        raise NotImplementedError

    def _grad(self, X, y):
        """
        Compute gradient of loss function.

        Parameters
        ----------
        X : array-like
            Input batch data.

        y : array-like
            Target batch values.

        Returns
        -------
        array-like
            Gradient vector.
        """
        raise NotImplementedError

    def _update_params(self, grad, iteration):
        """
        Update model parameters using computed gradient.

        Parameters
        ----------
        grad : array-like
            Gradient of loss function.

        iteration : int
            Current iteration index.

        Returns
        -------
        None
        """
        raise NotImplementedError

    def _extra_logs(self, X, y, grad, iter):
        """
        Provide optional additional logging information.

        Parameters
        ----------
        X : array-like
            Input data.

        y : array-like
            Target values.

        grad : array-like or None
            Gradient (if available).

        iter : int
            Current iteration index.

        Returns
        -------
        dict
            Additional log information.
        """
        return {}
