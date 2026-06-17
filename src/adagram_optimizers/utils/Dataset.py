import torch
from abc import ABC, abstractmethod
from torchvision import datasets, transforms
from ucimlrepo import fetch_ucirepo
import numpy as np

from sklearn.preprocessing import StandardScaler

import os
import pandas as pd
import torch
from sklearn.datasets import fetch_openml


class Dataset(ABC):
    """
    Abstract base class for generating synthetic datasets.
    """
    def __init__(self, n_samples, in_dim, out_dim=1, noise=0.1, seed=42):
        if in_dim < 2:
            raise ValueError("in_dim must be at least 2 for this logic.")
        
        self.n_samples = n_samples
        self.in_dim = in_dim
        self.noise = noise
        self.rng = np.random.RandomState(seed)
        # Using the provided weight creation strategy which is excellent
        # for creating a complex, non-axis-aligned problem.
        # self.true_weights_mask = self._create_weight_mask()
        self.true_weights_mask = self.rng.randn(in_dim, 1)

    def _create_weight_mask(self, anisotropy_ratio=1e4, seed=42):
        """Creates the ground-truth weight vector that defines the problem."""
        rng = np.random.default_rng(seed)

        # 1. Create a set of anisotropic "principal" weight magnitudes
        # The magnitudes are log-spaced to create a strong anisotropic effect.
        principal_weights = np.logspace(0, np.log10(anisotropy_ratio), self.in_dim)
        principal_weights *= rng.choice([-1, 1], size=self.in_dim)

        # 2. Create a random rotation matrix to introduce correlation
        random_matrix = rng.standard_normal(size=(self.in_dim, self.in_dim))
        rotation_matrix, _ = np.linalg.qr(random_matrix)

        # 3. Apply the rotation to create the final correlated weight vector
        correlated_weights = rotation_matrix @ principal_weights
        return correlated_weights.reshape(-1, 1)

    @abstractmethod
    def create_data(self):
        """
        Each child class must implement this to return X and y as NumPy arrays.
        """
        pass

    def _generate_labels(self, X):
        """
        Helper function to generate logistic regression labels from features and true weights.
        
        This is the main corrected part.
        """
        # 1. Calculate the linear combination (logits)
        logits = X @ self.true_weights_mask
        
        # 2. Apply the sigmoid function to get probabilities
        # A numerically stable sigmoid can be used, but for this problem,
        # large logit values are part of the design.
        sigmoid = lambda z: 1 / (1 + np.exp(-z))
        probabilities = sigmoid(logits)
        
        # 3. Generate binary outcomes from a Bernoulli distribution
        # This is done by comparing the probabilities to a uniform random number.
        y = (self.rng.uniform(size=probabilities.shape) < probabilities).astype(np.int64).flatten()
        return y

    def get_data_for_analysis(self):
        """
        A helper method to get X as a pandas DataFrame and y as a numpy array.
        """
        X_np, y_np = self.create_data()

        # 3. Create the DataFrame using the SCALED data and original feature names
        feature_names = [f'feature_{i}' for i in range(self.in_dim)]
        X_df = pd.DataFrame(X_np, columns=feature_names)
        # X_df = pd.DataFrame(X_np, columns=feature_names)
        return X_df, y_np

class CorrelatedAnisotropicDataset(Dataset):
    """
    A concrete implementation that generates a dataset with correlated features,
    which is the key to creating an anisotropic loss landscape for logistic regression.
    """
    # def _toeplitz_ar1_corr(self, rho: float) -> np.ndarray:
    #     # Toeplitz AR(1) correlation: R[i,j] = rho**|i-j|
    #     idx = np.arange(self.in_dim)
    #     return rho ** np.abs(idx[:, None] - idx[None, :])

    # def _safe_cholesky(self, R: np.ndarray, eps0: float = 1e-12, max_tries: int = 8) -> np.ndarray:
    #     eps = eps0
    #     for _ in range(max_tries):
    #         try:
    #             return np.linalg.cholesky(R)
    #         except np.linalg.LinAlgError:
    #             R = R + eps * np.eye(R.shape[0])
    #             eps *= 10.0
    #     # Eigenvalue clip fallback
    #     w, V = np.linalg.eigh(R)
    #     w = np.clip(w, 1e-10, None)
    #     R_spd = (V * w) @ V.T
    #     return np.linalg.cholesky(R_spd)

    def create_data(self):
        Z = self.rng.normal(loc=0.0, scale=1.0, size=(self.n_samples, self.in_dim))

        diag_vals = 1 + self.rng.uniform(0, 0.1, size=self.in_dim)
        sub_vals  = 1 + self.rng.uniform(0.1, 0.5, size=self.in_dim - 1)
    
        T = np.zeros((self.in_dim, self.in_dim), dtype=float)
        np.fill_diagonal(T, diag_vals.astype(float))

        T[np.arange(1, self.in_dim), np.arange(self.in_dim - 1)] = sub_vals.astype(float)
    
        X = Z @ T
    
        y = self._generate_labels(X)

        return torch.from_numpy(X).float(), torch.from_numpy(np.asarray(y)).long().flatten()
    
# --- Child Classes (Now Simpler) ---

class IsotropicDataset(Dataset):
    """Generates an Uncorrelated, Isotropic dataset."""
    def create_data(self):
        cov_matrix = np.eye(self.in_dim)
        X = self.rng.multivariate_normal(mean=np.zeros(self.in_dim), cov=cov_matrix, size=self.n_samples)
        y = self._generate_labels(X)
        return X, torch.tensor(y.reshape(-1), dtype=torch.long)


class AnisotropicDataset(Dataset):
    """Generates an Uncorrelated, Anisotropic dataset."""
    def create_data(self, anisotropy_ratio=1e3):
        variances = np.logspace(0, np.log10(anisotropy_ratio), self.in_dim)
        cov_matrix = np.diag(variances)
        X = self.rng.multivariate_normal(mean=np.zeros(self.in_dim), cov=cov_matrix, size=self.n_samples)
        y = self._generate_labels(X)
        return X, torch.tensor(y.reshape(-1), dtype=torch.long)



class StudentPerformanceDataset:
    """
    Loads the Student Performance dataset from the UCI repository.
    This is a REGRESSION task to predict the final grade (G3).
    """
    def get_data(self):
        """Downloads and caches the dataset, returns pandas DataFrame."""
        DATASET_NAME = "student_performance"
        DATA_FILENAME = os.path.join(DATA_DIR, f"{DATASET_NAME}.csv")
        
        if not os.path.exists(DATA_FILENAME):
            os.makedirs(DATA_DIR, exist_ok=True)
            print(f"Downloading {DATASET_NAME} dataset from UCI...")
            # Dataset ID for Student Performance is 320
            dataset = fetch_ucirepo(id=320)
            df = pd.concat([dataset.data.features, dataset.data.targets], axis=1)
            df.to_csv(DATA_FILENAME, index=False)
            
        return pd.read_csv(DATA_FILENAME)

    def create_data(self):
        """Processes the dataframe and returns PyTorch tensors."""
        df = self.get_data()

        # The target variable is 'G3' (final grade)
        X = df.drop("G3", axis=1)
        y = df["G3"]

        # 1. Explicitly identify categorical and numerical feature columns
        categorical_cols = X.select_dtypes(include=['object']).columns
        numerical_cols = X.select_dtypes(include=['number']).columns

        # 2. Apply one-hot encoding to the categorical columns
        # Using dtype=float ensures the new columns are floats, not booleans/integers
        X_categorical = pd.get_dummies(
            X[categorical_cols], 
            drop_first=True,  # Avoids multicollinearity
            dtype=float
        )

        # 3. Get the original numerical features
        X_numerical = X[numerical_cols]

        # 4. Concatenate the numerical features and the new one-hot encoded features
        X = pd.concat([X_numerical, X_categorical], axis=1)

        print("\nStudent Performance Shapes:")
        print("Features (X):", X.shape)
        print("Target   (y):", y.shape)

        # 5. Convert to float32 tensors. 
        # Since X is now guaranteed to be fully numeric, this will work reliably.
        return torch.tensor(X.values, dtype=torch.float32), torch.tensor(
            y.values.reshape(-1, 1), dtype=torch.float32
        )

DATA_DIR  = "./data"

class AIDSDataset:
    """
    Loads the AIDS Clinical Trials Group Study 175 dataset from UCI.
    This is set up for a REGRESSION task to predict 'time' (time to failure/censoring).
    """
    def get_data(self):
        """Downloads and caches the dataset, returns pandas DataFrame."""
        DATASET_NAME = "aids_clinical_trials"
        DATA_FILENAME = os.path.join(DATA_DIR, f"{DATASET_NAME}.csv")
        
        if not os.path.exists(DATA_FILENAME):
            os.makedirs(DATA_DIR, exist_ok=True)
            print(f"Downloading {DATASET_NAME} dataset from UCI...")
            # Dataset ID is 890
            dataset = fetch_ucirepo(id=890)
            df = pd.concat([dataset.data.features, dataset.data.targets], axis=1)
            df.to_csv(DATA_FILENAME, index=False)
            
        return pd.read_csv(DATA_FILENAME)

    def create_data(self):
        """Processes the dataframe and returns PyTorch tensors."""
        df = self.get_data()
        
        # Target for regression: 'time' to failure or censoring
        # Features: All other columns except the classification target ('cid')
        # 'pidnum' is already excluded by fetch_ucirepo as it's an ID column.
        
        # FIX: Remove 'pidnum' from the drop list as it's not in the DataFrame.
        X = df.drop(["time", "cid"], axis=1)
        y = df["time"]
        
        # Note: Features are already numeric (binary or integer). No one-hot encoding needed.
        
        print("\nAIDS Clinical Trials Shapes (Regression on 'time'):")
        print("Features (X):", X.shape)
        print("Target   (y):", y.shape)
    
        # Convert to float32 tensors
        return torch.tensor(X.values, dtype=torch.float32), torch.tensor(
            y.values.reshape(-1, 1), dtype=torch.float32
        )
        
class CommunitiesAndCrimeDataset:
    """
    Loads the Communities and Crime dataset from the UCI repository.
    This is a REGRESSION task to predict violent crime rates.
    This dataset contains missing values that need to be handled.
    """
    def get_data(self):
        """Downloads and caches the dataset, returns pandas DataFrame."""
        DATASET_NAME = "communities_and_crime"
        DATA_FILENAME = os.path.join(DATA_DIR, f"{DATASET_NAME}.csv")

        if not os.path.exists(DATA_FILENAME):
            os.makedirs(DATA_DIR, exist_ok=True)
            print(f"Downloading {DATASET_NAME} dataset from UCI...")
            # Dataset ID is 183
            dataset = fetch_ucirepo(id=183)
            df = pd.concat([dataset.data.features, dataset.data.targets], axis=1)
            df.to_csv(DATA_FILENAME, index=False)

        return pd.read_csv(DATA_FILENAME)

    def create_data(self):
        """Processes the dataframe and returns PyTorch tensors."""
        df = self.get_data()

        # 1. Replace the non-numeric '?' with NumPy's NaN
        df.replace('?', np.nan, inplace=True)

        # Drop non-predictive metadata columns
        cols_to_drop = ['state', 'county', 'community', 'communityname', 'fold']
        df = df.drop(columns=cols_to_drop)

        # The target variable is 'ViolentCrimesPerPop'
        X = df.drop('ViolentCrimesPerPop', axis=1)
        y = df['ViolentCrimesPerPop']

        # 2. Ensure all feature columns are numeric before imputation
        # This will convert object columns (due to '?') to float/int types
        X = X.apply(pd.to_numeric, errors='coerce')
        y = pd.to_numeric(y, errors='coerce')

        # 3. Impute the now-numeric NaN values with the median
        # Using DataFrame.fillna() is more efficient than a loop
        X.fillna(X.median(), inplace=True)
        # Also fill any potential missing values in the target variable
        y.fillna(y.median(), inplace=True)

        print("\nCommunities and Crime Shapes:")
        print("Features (X):", X.shape)
        print("Target   (y):", y.shape)

        # Convert to float32 tensors. This will now work without error.
        return torch.tensor(X.values, dtype=torch.float32), torch.tensor(
            y.values.reshape(-1, 1), dtype=torch.float32
        )
    

class HeartDataset:

    HEART_FILENAME = os.path.join(DATA_DIR, "heart.csv")
    HEART_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

    def get_heart_csv(self):
        if not os.path.exists(self.HEART_FILENAME):
            os.makedirs(DATA_DIR, exist_ok=True)
            print("Downloading Heart dataset...")
            cols = [  # columns from the UCI file
                "age",
                "sex",
                "cp",
                "trestbps",
                "chol",
                "fbs",
                "restecg",
                "thalach",
                "exang",
                "oldpeak",
                "slope",
                "ca",
                "thal",
                "target",
            ]
            df = pd.read_csv(self.HEART_URL, names=cols, na_values="?")
            # Clean missing values
            df = df.dropna().reset_index(drop=True)
            df.to_csv(self.HEART_FILENAME, index=False)
        return pd.read_csv(self.HEART_FILENAME)

    def get_data_for_analysis(self):
        """
        A helper method to get X and y as pandas objects, ideal for analysis.
        """
        df = self.get_heart_csv()
        
        # CORRECTED: X should be all columns except the target 'class'
        X = df.drop(columns=['target'])
        
        # CORRECTED: y is the 'class' column
        y, uniques = pd.factorize(df['target']) # Factorize turns strings ('+','-') into (0,1)
        
        # One-hot encode categorical features within X
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        X_processed = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
        return X_processed, y

    def create_data(self):
        df = self.get_heart_csv()
        # Make binary labels: according to UCI, 'target'==0 as healthy, >0 as unhealthy
        X = df.iloc[:, :-1].values.astype(float)
        y = (df["target"].values > 0).astype(int).astype(int)
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


#########################################
# Australian Credit (OpenML or LIBSVM)  #
#########################################


class AustralianCreditDataset:
    def get_csv(self):
        DATA_FILENAME = os.path.join(DATA_DIR, "australian.csv")
        if not os.path.exists(DATA_FILENAME):
            os.makedirs(DATA_DIR, exist_ok=True)
            print("Downloading dataset from OpenML...")
            data = fetch_openml(data_id=40509, as_frame=True)
            df = data.frame
            df.to_csv(DATA_FILENAME, index=False)
        return pd.read_csv(DATA_FILENAME)
    
    def get_data_for_analysis(self):
        """
        A helper method to get X and y as pandas objects, ideal for analysis.
        """
        df = self.get_csv()
        
        # CORRECTED: X should be all columns except the target 'class'
        X = df.iloc[:, 1:]
        
        # CORRECTED: y is the 'class' column
        y = pd.factorize(df.iloc[:, 1]) # Factorize turns strings ('+','-') into (0,1)
        
        # One-hot encode categorical features within X
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        X_processed = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
        
        return X_processed, y

    def create_data(self):
        df = self.get_csv()
        X = df.iloc[:, 1:]
        X = pd.get_dummies(X)  # ensures all features are numeric

        y = df.iloc[:, 1]
        y, uniques = pd.factorize(y)  # makes target integer 0, 1, ...

        print(X.shape)
        print(y.shape)
        # CORRECTION: specify dtype=torch.long for classification targets!
        return torch.tensor(X.values, dtype=torch.float32), torch.tensor(
            y, dtype=torch.long
        )


############################
# Splice Dataset (UCI)     #
############################


class SpliceDataset:
    def __init__(self, n_samples=None):
        self.n_samples = n_samples
        self.splicedata = fetch_ucirepo(id=69)

    def get_csv(self):
        # You can customize this to return a DataFrame for your pipeline logic
        X = self.splicedata.data.features
        y = self.splicedata.data.targets
        df = X.copy()
        df['target'] = y
        if self.n_samples:
            df = df.iloc[:self.n_samples]
        return df
    
    def get_data_for_analysis(self):
        """
        A helper method to get X and y as pandas objects, ideal for analysis.
        """
        df = self.get_csv()
        
        # CORRECTED: X should be all columns except the target 'class'
        X = df.drop(columns=['target'])
        
        # CORRECTED: y is the 'class' column
        y, uniques = pd.factorize(df['target']) # Factorize turns strings ('+','-') into (0,1)
        
        # One-hot encode categorical features within X
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        X_processed = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
        
        return X_processed, y

    def create_data(self):
        df = self.get_csv()
        # column 'target' is the label, rest are features
        X = df.drop(columns=['target'])
        X = pd.get_dummies(X)  # One-hot for categorical features (usually needed for tabular data)
        y, uniques = pd.factorize(df['target'])
        print(X.shape)
        print(y.shape)
        return torch.tensor(X.values, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


class SparseDataset(Dataset):
    def __init__(
        self,
        n_samples,
        in_dim,
        out_dim=1,
        noise=0.1,
        sparsity_rate_1=0.05,
        sparsity_rate_2=0.15,
        seed=42,
        if_class=True,
    ):
        super().__init__(n_samples, in_dim, out_dim, noise)
        self.sparsity_rate_1 = sparsity_rate_1
        self.sparsity_rate_2 = sparsity_rate_2
        self.seed = seed
        self.if_class = if_class

    def create_data(self):
        if self.if_class:
            X, y = self.create_binary_data()
        else:
            X, y = self.create_reg_data()
        return X, y

    def create_scaled_binary_data_matrix(self, scaling_factors=None):
        """
        Create scaled dataset using explicit diagonal matrix multiplication.
        """
        X, y = self.create_binary_data()

        if scaling_factors is None:
            scaling_factors = torch.logspace(-2.0, 3.0, self.in_dim)

        # Create diagonal matrix V
        V = torch.diag(scaling_factors)

        # Apply scaling: X_scaled = X @ V
        X_scaled = X @ V

        return X, X_scaled, y

    def create_binary_data(self):
        """
        Create toy data where only 2-3 features are truly important,
        and these important features are sparse (mostly zero).

        Returns:
            tuple: (X, y) where X is sparse input data and y is binary target
        """

        if self.seed:
            torch.manual_seed(self.seed)
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
            5.0 * X[:, 0]  # Sparse feature 1 (high weight)
            + 1.5 * X[:, 1]  # Sparse feature 2 (medium weight)
            + 0.3
            * X[:, 2 : min(5, self.in_dim)].sum(dim=1)  # Dense features (low weight)
            # + torch.randn(self.n_samples) * self.noise
        )  # Noise

        # Convert to binary classification

        y = (y > y.median()).long()

        return X, y

    def create_reg_data(self):
        """
        Create toy data where only 2-3 features are truly important,
        and these important features are sparse (mostly zero).

        Returns:
            tuple: (X, y) where X is sparse input data and y is binary target
        """

        if self.seed:
            torch.manual_seed(self.seed)
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

        Y = torch.zeros((self.n_samples, self.out_dim))
        for i in range(self.out_dim):
            Y[:, i] = (
                2.0 * X[:, 0]
                + 1.5 * X[:, 1]
                + 0.3 * X[:, 2 : min(5, self.in_dim)].sum(axis=1)
                + torch.randn(self.n_samples) * self.noise
            )
        return X, Y


class CorrelatedDataset(Dataset):
    def __init__(
        self,
        n_samples,
        in_dim,
        out_dim=1,
        seed=100,
        noise=0.1,
        correlation_strength=0.7,
    ):
        super().__init__(n_samples, in_dim, out_dim, noise)
        self.correlation_strength = correlation_strength

    def create_scaled_binary_data_matrix(self):
        pass

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
    def __init__(self, n_samples, in_dim, out_dim=1, seed=100, noise=0.1):
        super().__init__(n_samples, in_dim, out_dim, noise)

    def create_scaled_binary_data_matrix(self):
        pass

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

    def create_scaled_binary_data_matrix(self):
        pass

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


class MNISTDataWrapper:
    def create_data(self):
        # Returns full training set for compatibility
        transform = transforms.Compose([transforms.ToTensor()])
        train_set = datasets.MNIST(
            root="./data", train=True, download=True, transform=transform
        )
        X = train_set.data.view(-1, 28 * 28).float() / 255.0
        y = train_set.targets
        return X, y
