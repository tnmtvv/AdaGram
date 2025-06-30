import torch
from abc import ABC, abstractmethod


class Dataset(ABC):
    def __init__(self, n_samples, in_dim, out_dim=1, noise=0.1):
        self.n_samples = n_samples
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.noise = noise

    @abstractmethod
    def create_data(self):
        """
        Create dataset with consistent interface.

        Returns:
            tuple: (X, y) where X is input data and y is target data
        """
        pass


class SparseDataset(Dataset):
    def __init__(
        self,
        n_samples,
        in_dim,
        out_dim=1,
        noise=0.1,
        sparsity_rate_1=0.05,
        sparsity_rate_2=0.15,
    ):
        super().__init__(n_samples, in_dim, out_dim, noise)
        self.sparsity_rate_1 = sparsity_rate_1
        self.sparsity_rate_2 = sparsity_rate_2

    def create_data(self):
        """
        Create toy data where only 2-3 features are truly important,
        and these important features are sparse (mostly zero).

        Returns:
            tuple: (X, y) where X is sparse input data and y is binary target
        """
        X = torch.zeros(self.n_samples, self.in_dim)

        # Feature 1: Very sparse but highly predictive
        sparse_mask_1 = torch.rand(self.n_samples) < self.sparsity_rate_1
        X[sparse_mask_1, 0] = torch.randn(sparse_mask_1.sum()) * 3

        # Feature 2: Moderately sparse but important
        sparse_mask_2 = torch.rand(self.n_samples) < self.sparsity_rate_2
        X[sparse_mask_2, 1] = torch.randn(sparse_mask_2.sum()) * 2

        # Features 3-5: Dense but less important (noise features)
        if self.in_dim > 5:
            X[:, 2:5] = torch.randn(self.n_samples, min(3, self.in_dim - 2)) * 0.5
            # Remaining features: Pure noise
            X[:, 5:] = torch.randn(self.n_samples, self.in_dim - 5) * 0.1
        else:
            X[:, 2:] = torch.randn(self.n_samples, self.in_dim - 2) * 0.5

        # Create target: heavily dependent on sparse features
        y = (
            2.0 * X[:, 0]  # Sparse feature 1 (high weight)
            + 1.5 * X[:, 1]  # Sparse feature 2 (medium weight)
            + 0.3
            * X[:, 2 : min(5, self.in_dim)].sum(dim=1)  # Dense features (low weight)
            + torch.randn(self.n_samples) * self.noise
        )  # Noise

        # Convert to binary classification
        y = (y > y.median()).long()

        return X, y


class CorrelatedDataset(Dataset):
    def __init__(
        self, n_samples, in_dim, out_dim=1, noise=0.1, correlation_strength=0.7
    ):
        super().__init__(n_samples, in_dim, out_dim, noise)
        self.correlation_strength = correlation_strength

    def create_data(self):
        """
        Generate synthetic data with correlated input features.

        Returns:
            tuple: (X, y) where X is correlated input data and y is target data
        """
        # Create correlation matrix for input features
        correlation_matrix = torch.full(
            (self.in_dim, self.in_dim), self.correlation_strength
        )
        correlation_matrix.fill_diagonal_(1.0)

        # Generate correlated input data using multivariate normal distribution
        mean = torch.zeros(self.in_dim)
        X_correlated = torch.distributions.MultivariateNormal(
            mean, correlation_matrix
        ).sample((self.n_samples,))

        # Scale and shift to desired range [0, 10]
        X = (
            (X_correlated - X_correlated.min())
            / (X_correlated.max() - X_correlated.min())
            * 10
        )

        # True weights and biases
        true_weights = torch.randn(self.in_dim, self.out_dim) * 2
        true_bias = torch.randn(self.out_dim) * 1

        # Linear transformation with noise
        y = (
            X @ true_weights
            + true_bias
            + self.noise * torch.randn(self.n_samples, self.out_dim)
        )

        return X, y


class LinearDataset(Dataset):
    def __init__(self, n_samples, in_dim, out_dim=1, noise=0.1):
        super().__init__(n_samples, in_dim, out_dim, noise)

    def create_data(self):
        """
        Generate simple linear dataset.

        Returns:
            tuple: (X, y) where X is input data and y is linear target
        """
        X = torch.randn(self.n_samples, self.in_dim)

        # True weights and bias
        true_weights = torch.randn(self.in_dim, self.out_dim)
        true_bias = torch.randn(self.out_dim)

        # Linear transformation with noise
        y = (
            X @ true_weights
            + true_bias
            + self.noise * torch.randn(self.n_samples, self.out_dim)
        )

        return X, y


class NonLinearDataset(Dataset):
    def __init__(self, n_samples, in_dim, out_dim=1, noise=0.1):
        super().__init__(n_samples, in_dim, out_dim, noise)

    def create_data(self):
        """
        Generate non-linear dataset with polynomial features.

        Returns:
            tuple: (X, y) where X is input data and y is non-linear target
        """
        X = torch.randn(self.n_samples, self.in_dim)

        # Non-linear transformation: quadratic + interaction terms
        y = (
            X[:, 0] ** 2
            + X[:, 1] * X[:, 0]
            + torch.sin(X[:, 0])
            + 0.5 * X[:, 1:].sum(dim=1)
            + self.noise * torch.randn(self.n_samples)
        )

        if self.out_dim > 1:
            y = y.unsqueeze(1).repeat(1, self.out_dim)

        return X, y
