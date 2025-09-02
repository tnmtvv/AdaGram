import os
import random
import time
import numpy as np
import pandas as pd
from itertools import product

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from typing import Optional
import copy
import cProfile

import libcontext

from src.utils.Dataset import (
    SparseDataset,
    MNISTDataWrapper,
    SpliceDataset,
    HeartDataset,
    AustralianCreditDataset,
    CorrelatedAnisotropicDataset,
    IsotropicDataset, 
    AnisotropicDataset,
    CommunitiesAndCrimeDataset,
    StudentPerformanceDataset,
    AIDSDataset
)

from src.AdagramSVD import AdaGramFR
from src.AdagramVanilla import AdaGramVanilla
from src.AdagramPS import AdaGramPS
from src.Shampoo import Shampoo
from src.DiagAdagrad import CustomAdaGrad
from src.FullAdagrad import FullAdaGrad
from src.Kate import KATE

from src.utils.models import (
    LinearRegressionModel,
    SimpleClassifier,
)

from src.utils.Config import Config
from src.utils.Trainer import Trainer

class ExperimentRunner:
    """Orchestrates the overall experiment, delegating training to the Trainer class."""
    
    def __init__(self, config: Config, training_mode: str):
        self.config = config
        self.seeds = self.config.get("experiment.seed", [42])
        self._setup_directories()
        self.results = []  # No comma
        self.training_mode = training_mode  # No comma
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

    # ... [Keep seed_everything, _setup_directories, _get_dataset, _get_model, _get_optimizer as they were] ...
    def seed_everything(self, seed: int):
        """Set all random seeds for reproducibility"""
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _setup_directories(self):
        """Create necessary directories"""
        for dir_key in ["results_dir", "plots_dir", "gradients_dir"]:
            directory = self.config.get(f"output.{dir_key}")
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
    def _get_dataset(self, in_dim: int, out_dim: int, seed: int):
        """Create dataset based on configuration"""
        dataset_type = self.config.get("dataset.type")
        n_samples = self.config.get("data.n_samples")

        if dataset_type == "SparseDataset":
            if_class = self.config.get("dataset.sparse_config.if_class", False)
            return SparseDataset(
                n_samples=n_samples,
                in_dim=in_dim,
                out_dim=out_dim,
                seed=seed,
                if_class=if_class,
            )
        
        if dataset_type == "IsotropicDataset":
            return IsotropicDataset(
                n_samples=n_samples,
                in_dim=in_dim,
                out_dim=out_dim,
                seed=seed,
            )
        if dataset_type == "AnisoDataset":
            return AnisotropicDataset(
                n_samples=n_samples,
                in_dim=in_dim,
                out_dim=out_dim,
                seed=seed,
            )
        if dataset_type == "CorrelatedAnisotropicDataset":
            return CorrelatedAnisotropicDataset(
                n_samples=n_samples,
                in_dim=in_dim,
                out_dim=out_dim,
                seed=seed,
            )
        elif dataset_type == "MNIST":
            return MNISTDataWrapper()
        elif dataset_type == "AU":
            return AustralianCreditDataset()
        elif dataset_type == "Heart":
            return HeartDataset()
        elif dataset_type == "Splice":
            return SpliceDataset()
        
        elif dataset_type == "Crime":
            return CommunitiesAndCrimeDataset()
        elif dataset_type == "Aids":
            return AIDSDataset()
        elif dataset_type == "Student":
            return StudentPerformanceDataset()
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")


    def _get_model(self, task_name: str, in_dim: int, out_dim: int, model_seed:int):
        """Create model based on task and configuration"""
        # model_seed = self.config.get(f"models.{task_name}.model_seed", 100)
        if not model_seed:
            raise ValueError("No noise_std")

        if task_name == "BinClass":
            return SimpleClassifier(
                input_dim=in_dim, output_dim=out_dim, seed=model_seed
            )
        elif task_name == "LinReg":
            return LinearRegressionModel(
                dim_in=in_dim, dim_out=out_dim, seed=model_seed
            )
        else:
            raise ValueError(f"Unknown task: {task_name}")


    def _get_loss_function(self, task_name: str):
        """Get loss function based on task"""
        loss_name = self.config.get(f"models.{task_name}.loss_function")

        if loss_name == "CrossEntropyLoss":
            return nn.CrossEntropyLoss()
        elif loss_name == "MSELoss":
            return nn.MSELoss()
        else:
            raise ValueError(f"Unknown loss function: {loss_name}")


    def _get_optimizer(
        self,
        opt_name: str,
        params,
        lr: float,
        eps: float,
        testing: bool = False,
        max_rank: Optional[int] = None,
        alpha = None,
        task: str = "LinReg",
    ):
        
        """Create optimizer based on configuration"""
        
        optimizer_map = {
            "AdaGramPS": lambda: AdaGramPS(
                params=params,
                lr=lr,
                max_rank=max_rank,
                task=task,
                eps=eps,
                alpha = alpha,
                enable_logging= testing,
                save_dir="matrix_G",
            ),
            "AdaGramPS_fullrank": lambda: AdaGramPS(
                params=params,
                lr=lr,
                max_rank=None,
                task=task,
                eps=eps,
                alpha = alpha,
                enable_logging= testing,
                save_dir="matrix_G",
            ),
            "AdaGramFR_svd": lambda: AdaGramFR(
                params, lr=lr, max_rank=max_rank, task=task, eps=eps, enable_logging= testing,
            ),
            "AdaGramFR_nosvd": lambda: AdaGramFR(
                params, lr=lr, max_rank=max_rank, eps=eps, enable_logging=testing,
            ),
            "KATE": lambda: KATE(params, lr=lr, eps=eps),
            "Torch_Adagrad": lambda: torch.optim.Adagrad(params, lr=lr, eps=eps),
            "Shampoo": lambda: Shampoo(params, lr=lr, eps=eps),
            "FullAdaGrad": lambda: FullAdaGrad(params=params, lr=lr, eps=eps),
            "AdaGram": lambda: AdaGramVanilla(params, lr=lr, eps=eps, enable_logging= testing,),
            "Vanilla_SGD": lambda: torch.optim.SGD(params, lr=lr),
        }

        return optimizer_map[opt_name]()
    
    def run_experiment(self):
        """Run the main experiment loop."""
        # ... [Get configs for seeds, dims, tasks etc.] ...
        trainer = Trainer(self.config, self.device, self.results, mode=self.training_mode)

        in_dims = self.config.get("data.in_dims")
        out_dims = self.config.get("data.out_dims")
        enabled_tasks = self.config.get("tasks.enabled_tasks")

        if not (enabled_tasks or in_dims or out_dims):
            raise ValueError

        task_name = enabled_tasks[0]
        for seed in self.seeds:
            ds = self._get_dataset(in_dims[0], out_dims[0], seed)
            base_model = self._get_model(task_name, in_dims[0], out_dims[0], model_seed=seed).to(self.device)

            X, y = ds.create_data()

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=self.config.get("data.test_size"),
                random_state=42,
            )

            scaler = StandardScaler()
            X_train = torch.tensor(scaler.fit_transform(X_train), dtype=torch.float32)
            X_test =  torch.tensor(scaler.transform(X_test), dtype=torch.float32)

            X_train = X_train.to(self.device)
            X_test = X_test.to(self.device)
            y_train = y_train.to(self.device)
            y_test = y_test.to(self.device)

            print(f"X_train shape: {X_train.shape}")
            print(f"y_train shape: {y_train.shape}")

            criterion = self._get_loss_function(task_name)

            for opt_name, opt_config in self.config.optimizers.items():
                if not opt_config.get("enabled"): continue
                print(f"Running optimizer: {opt_name}")
                
                # Simplified hyperparameter iteration
                hparams_grid = self._create_hparam_grid(opt_config)

                for params in hparams_grid:
                    try:
                        model = copy.deepcopy(base_model)
                        optimizer = self._get_optimizer(
                            opt_name.split(' ')[0], model.parameters(),
                            lr=params['lr'], eps=params['eps'],
                            testing=opt_config.get("testing"), alpha=params['alpha'],
                            max_rank=params['rank'], task=task_name
                        )

                        # Delegate the entire training process to the trainer
                        trainer.train(
                            model, optimizer, criterion, X_train, y_train, X_test, y_test,
                            opt_name=opt_name, data_seed=seed, **params
                        )
                    except Exception as e:
                        print(f"Exception occurred for {opt_name} with params {params}: {e}")

        df = pd.DataFrame(self.results)
        if not df.empty:
            df["loss"] = df["loss"].astype(float)
        return df

    def _create_hparam_grid(self, opt_config):
        """
        Helper to create a grid of hyperparameters to iterate over. This version is
        robust to `None` values in the configuration.
        """
        # Helper to ensure the value is a list, providing a default if None.
        def get_list(key, default):
            val = opt_config.get(key)
            if val is None:
                return default
            # Ensure single values are wrapped in a list for iteration
            return val if isinstance(val, list) else [val]

        # Build the parameter dictionary with robust list conversion
        params = {
            'lr': [float(lr) for lr in get_list("learning_rates", [1.0])],
            'eps': [float(e) for e in get_list("eps", [1e-8])],
            'batch_size': get_list("batch_size", [32]),
            'rank': get_list("ranks", [None]) if opt_config.get("requires_rank") else [None],
            'alpha': get_list("alphas", [None]) if opt_config.get("requires_rank") else [None]
        }

        # The rest of the function remains the same
        keys = params.keys()
        vals = params.values()

        for instance in product(*vals):
            yield dict(zip(keys, instance))

