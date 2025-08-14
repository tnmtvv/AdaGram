import os
import random
import time
import numpy as np
import pandas as pd

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

class ExperimentRunner:
    """Main experiment runner class"""

    def __init__(self, config: Config):
        self.config = config
        if self.config is None or self.config.config is None:
            raise ValueError("Configuration failed to load")

        seed = self.config.get("experiment.seed", 42)
        if not seed:
            raise ValueError("seed is not defined")
        self.seed_everything(int(seed))
        self._setup_directories()
        self.results = []

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

    # ------ Getters -------

    def seed_everything(self, seed: int):
        """Set all random seeds for reproducibility"""
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
        torch.backends.cudnn.benchmark = False # Changed from True to False 

    def _setup_directories(self):
        """Create necessary directories"""
        directories = [
            self.config.get("output.results_dir"),
            self.config.get("output.plots_dir"),
            self.config.get("output.gradients_dir"),
        ]

        for directory in directories:
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

    def _get_model(self, task_name: str, in_dim: int, out_dim: int):
        """Create model based on task and configuration"""
        model_seed = self.config.get(f"models.{task_name}.model_seed", 100)
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
            "Torch_Adagrad": lambda: CustomAdaGrad(params, lr=lr, eps=eps),
            "Shampoo": lambda: Shampoo(params, lr=lr, eps=eps),
            "FullAdaGrad": lambda: FullAdaGrad(params=params, lr=lr, eps=eps),
            "AdaGram": lambda: AdaGramVanilla(params, lr=lr, eps=eps, enable_logging= testing,),
            "Vanilla_SGD": lambda: torch.optim.SGD(params, lr=lr),
        }

        return optimizer_map[opt_name]()



    # ------ Logic -------

    def _calculate_metrics(self, model, criterion, X, y, if_class):
        """Calculates loss and accuracy for a given dataset."""
        with torch.no_grad():
            y_pred = model(X)
            loss = criterion(y_pred, y)
            
            accuracy = 0
            if if_class:
                y_pred_probs = torch.softmax(y_pred, dim=1)
                predicted_labels = torch.argmax(y_pred_probs, dim=1)
                accuracy = (predicted_labels == y).float().mean().item()
                
        return loss.item(), accuracy

    def _log_results(self, epoch, elapsed_time, metrics, opt_name, lr, batch_size, r, eps,  data_seed):
        """Logs training and testing metrics."""
        train_loss, train_acc, test_loss, test_acc = metrics
        r_in_name = f" rank {r}" if r is not None else ""
        
        common_info = {
            "optimizer": opt_name + r_in_name,
            "lr": lr,
            "rank": r,
            "eps": eps,
            "batch_size": batch_size,
            "data_seed": data_seed,
            "epoch_time": elapsed_time,
            "avg_epoch_time": elapsed_time / (epoch + 1) if epoch > -1 else 0,
        }

        self.results.extend([
            {**common_info, "epoch": epoch + 1, "loss": test_loss, "accuracy": test_acc, "mode": "test"},
            {**common_info, "epoch": epoch + 1, "loss": train_loss, "accuracy": train_acc, "mode": "train"},
        ])

    def _train_model_stochastic(
        self,
        model,
        optimizer,
        criterion,
        X_train,
        y_train,
        X_test,
        y_test,
        opt_name,
        lr,
        batch_size,
        eps,
        stop_condition,
        r=None,
        data_seed=None,
    ):
        """A generalized stochastic training function."""
        # --- Configuration ---
        shuffle = self.config.get("training.shuffle")
        if_class = self.config.get("dataset.sparse_config.if_class")

        if data_seed:
            self.seed_everything(data_seed)

        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

        start_time = time.time()
        epoch = 0

        # --- Initial Metrics (before training) ---
        initial_metrics = (
            *self._calculate_metrics(model, criterion, X_train, y_train, if_class),
            *self._calculate_metrics(model, criterion, X_test, y_test, if_class),
        )
        self._log_results(-1, 0, initial_metrics, opt_name, lr, batch_size, r, eps, data_seed)
        
        # --- Training Loop ---
        while stop_condition(epoch, start_time):
            model.train()
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                y_pred = model(batch_X)
                loss = criterion(y_pred, batch_y)
                loss.backward()
                
                if r:
                    optimizer.step(epoch)
                else:
                    optimizer.step()

            # --- Post-Epoch Metrics and Logging ---
            elapsed_time = time.time() - start_time
            epoch_metrics = (
                *self._calculate_metrics(model, criterion, X_train, y_train, if_class),
                *self._calculate_metrics(model, criterion, X_test, y_test, if_class),
            )
            self._log_results(epoch, elapsed_time, epoch_metrics, opt_name, lr, batch_size, r, eps, data_seed)
            
            epoch += 1

        # --- Final Evaluation ---
        model.eval()
        final_test_loss, _ = self._calculate_metrics(model, criterion, X_test, y_test, if_class)
        return final_test_loss

    def train_model_stochastic_epochs(
        self,
        model,
        optimizer,
        criterion,
        X_train,
        y_train,
        X_test,
        y_test,
        opt_name,
        lr,
        batch_size,
        eps,
        r=None,
        data_seed=None,
    ):
        """Train model for a fixed number of epochs."""
        num_epochs = self.config.get("training.num_epochs")
        if not num_epochs:
            raise ValueError("num_epochs cannot be None")

        stop_condition = lambda epoch, start_time: epoch < num_epochs
        
        return self._train_model_stochastic(
            model, optimizer, criterion, X_train, y_train, X_test, y_test,
            opt_name, lr, batch_size, eps, stop_condition, r, data_seed
        )

    def train_model_stochastic_time(
        self,
        model,
        optimizer,
        criterion,
        X_train,
        y_train,
        X_test,
        y_test,
        opt_name,
        lr,
        batch_size,
        eps,
        time_budget=60,
        r=None,
        data_seed=None,
    ):
        """Train model for a given time budget."""
        stop_condition = lambda epoch, start_time: (time.time() - start_time) < time_budget

        return self._train_model_stochastic(
            model, optimizer, criterion, X_train, y_train, X_test, y_test,
            opt_name, lr, batch_size, eps, stop_condition, r, data_seed
        )

    def run_experiment(self):
        """Run the main experiment"""

        data_seeds = self.config.get("data.data_seeds")
        in_dims = self.config.get("data.in_dims")
        out_dims = self.config.get("data.out_dims")
        enabled_tasks = self.config.get("tasks.enabled_tasks")

        if not data_seeds or not in_dims or not out_dims:
            raise ValueError("data_seeds or dimensions are not defined")

        for data_seed in data_seeds:
            ds = self._get_dataset(in_dims[0], out_dims[0], data_seed)
            if not enabled_tasks:
                raise ValueError

            for task_name in enabled_tasks:
                base_model = self._get_model(task_name, in_dims[0], out_dims[0]).to(self.device)

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
                    if not opt_config["enabled"]:
                        continue
                
                    learning_rates = opt_config.get("learning_rates")
                    batches = opt_config.get("batch_size")
                    epsilons = opt_config.get("eps")

                    epsilons = [float(e) for e in epsilons]
                    ranks = opt_config.get("ranks")

                    print(f"Running optimizer: {opt_name}")

                    if not learning_rates or not epsilons or not batches:
                        raise ValueError("lrs or ranks are not defined")
                    if not ranks:
                        ranks = [None]
                        rank = None
                    
                    for bs in batches:
                        for lr in learning_rates:
                            for eps in epsilons:
                                if opt_config["requires_rank"]:
                                    # if len(ranks) > 0:
                                    for rank in ranks:
                                        try:
                                            model = copy.deepcopy(base_model)
                                            print("rank", rank)
                                            optimizer = self._get_optimizer(
                                                opt_name,
                                                model.parameters(),
                                                lr=lr,
                                                eps=eps,
                                                testing=opt_config["testing"],
                                                max_rank=rank,
                                                task=task_name,
                                            )

                                            self.train_model_stochastic_epochs(
                                                model=model,
                                                optimizer=optimizer,
                                                criterion=criterion,
                                                X_train=X_train,
                                                y_train=y_train,
                                                X_test=X_test,
                                                y_test=y_test,
                                                opt_name=opt_name,
                                                lr=lr,
                                                batch_size=bs,
                                                eps=eps,
                                                r=rank,
                                                data_seed=data_seed,
                                            )
                                        except Exception as e:
                                            print("Exception ocсured: ", e)
                                            continue
                                else:
                                    try:
                                        model = copy.deepcopy(base_model).to(self.device)
                                        optimizer = self._get_optimizer(
                                            opt_name, model.parameters(), lr, eps
                                        )

                                        self.train_model_stochastic_epochs(
                                            model=model,
                                            optimizer=optimizer,
                                            criterion=criterion,
                                            X_train=X_train,
                                            y_train=y_train,
                                            X_test=X_test,
                                            y_test=y_test,
                                            opt_name=opt_name,
                                            eps=eps,
                                            lr=lr,
                                            batch_size=bs,
                                            data_seed=data_seed,
                                        )
                                    except Exception as e:
                                        print("Exception ocсured: ", e)
                                        continue

        df = pd.DataFrame(self.results)
        df["loss"] = df["loss"].astype(float)

        return df