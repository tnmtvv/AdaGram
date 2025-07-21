import os
import random
import time
import yaml
from tqdm import tqdm
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sys
import libcontext
import glob
from typing import Optional
import re
import copy

from src.adagram_fixed_rank import AdaGramFR
from src.adagram_vanilla import AdaGramVanilla
from src.adagram_projector_splitting import AdaGramPS
from src.shampoo import Shampoo
from src.full_G import FullAdaGrad
from src.Kate import KATE
from src.utils.dataset import SparseDataset, CorrelatedDataset, LinearDataset
from src.utils.models import (
    LinearRegressionModel,
    MultiClassLogisticRegressionModel,
    SimpleClassifier,
)


class Config:
    """Configuration management class"""

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)

    def get(self, key_path: str, default=None):
        """Get configuration value using dot notation (e.g., 'training.num_epochs')"""
        keys = key_path.split(".")
        value = self.config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value

    def __getattr__(self, name):
        return self.config.get(name)


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
        self.setup_directories()
        self.results = []

    def seed_everything(self, seed: int):
        """Set all random seeds for reproducibility"""
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

    def setup_directories(self):
        """Create necessary directories"""
        directories = [
            self.config.get("output.results_dir"),
            self.config.get("output.plots_dir"),
            self.config.get("output.gradients_dir"),
        ]

        for directory in directories:
            if directory and not os.path.exists(directory):
                os.makedirs(directory)

    def get_dataset(self, in_dim: int, out_dim: int, seed: int):
        """Create dataset based on configuration"""
        dataset_type = self.config.get("dataset.type")
        n_samples = self.config.get("data.n_samples")

        if dataset_type == "SparseDataset":
            if_class = self.config.get("dataset.sparse_config.if_class", False)
            if if_class is None:
                raise ValueError("No if class")
            return SparseDataset(
                n_samples=n_samples,
                in_dim=in_dim,
                out_dim=out_dim,
                seed=seed,
                if_class=if_class,
            )
        elif dataset_type == "CorrelatedDataset" and seed:
            correlation = self.config.get("dataset.correlated_config.correlation", 0.8)
            if not correlation:
                raise ValueError("No correlation")
            return CorrelatedDataset(
                n_samples=n_samples,
                in_dim=in_dim,
                out_dim=out_dim,
                seed=seed,
                correlation_strength=correlation,
            )
        elif dataset_type == "LinearDataset":
            noise_std = self.config.get("dataset.linear_config.noise_std", 0.1)
            if not noise_std:
                raise ValueError("No noise_std")
            return LinearDataset(
                n_samples=n_samples,
                in_dim=in_dim,
                out_dim=out_dim,
                seed=seed,
                noise=noise_std,
            )
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")

    def get_model(self, task_name: str, in_dim: int, out_dim: int):
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

    def get_loss_function(self, task_name: str):
        """Get loss function based on task"""
        loss_name = self.config.get(f"models.{task_name}.loss_function")

        if loss_name == "CrossEntropyLoss":
            return nn.CrossEntropyLoss()
        elif loss_name == "MSELoss":
            return nn.MSELoss()
        else:
            raise ValueError(f"Unknown loss function: {loss_name}")

    def get_optimizer(
        self,
        opt_name: str,
        params,
        lr: float,
        eps: float,
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
                save_dir="matrix_G",
            ),
            "AdaGramFR_svd": lambda: AdaGramFR(
                params, lr=lr, max_rank=max_rank, task=task, eps=eps
            ),
            "AdaGramFR_nosvd": lambda: AdaGramFR(
                params, lr=lr, max_rank=max_rank, eps=eps
            ),
            "KATE": lambda: KATE(params, lr=lr, eps=eps),
            "Torch_Adagrad": lambda: torch.optim.Adagrad(params, lr=lr, eps=eps),
            "Shampoo": lambda: Shampoo(params, lr=lr, eps=eps),
            "FullAdaGrad": lambda: FullAdaGrad(params=params, lr=lr, eps=eps),
            "AdaGram": lambda: AdaGramVanilla(params, lr=lr, eps=eps),
            "Vanilla_SGD": lambda: torch.optim.SGD(params, lr=lr, eps=eps),
        }

        return optimizer_map[opt_name]()

    def train_model_stochastic(
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
        eps,
        X_type,
        r=None,
        data_seed=None,
        task_name=None,
    ):
        """Train model using stochastic gradient descent"""

        num_epochs = self.config.get("training.num_epochs")
        batch_size = self.config.get("training.batch_size")
        shuffle = self.config.get("training.shuffle")
        use_tqdm = self.config.get("training.use_tqdm")
        grad_save_dir = self.config.get("output.gradients_dir")

        if data_seed:
            self.seed_everything(data_seed)

        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

        if not num_epochs:
            raise ValueError("num_epochs is none")
        epoch_iterator = tqdm(range(num_epochs)) if use_tqdm else range(num_epochs)
        time_start = time.time()

        for epoch in epoch_iterator:
            model.train()
            start_epoch = time.time()
            epoch_loss = 0.0
            num_batches = 0

            if epoch == 0:
                y_pred_train = model(X_train)
                y_pred_test = model(X_test)

                train_loss = criterion(y_pred_train, y_train)
                test_loss = criterion(y_pred_test, y_test)

                with torch.no_grad():
                    # Apply softmax to convert logits to probabilities
                    y_pred_probs_train = torch.softmax(y_pred_train, dim=1)
                    y_pred_probs_test = torch.softmax(y_pred_test, dim=1)

                    # Get predicted class labels
                    predicted_labels_train = torch.argmax(y_pred_probs_train, dim=1)
                    predicted_labels_test = torch.argmax(y_pred_probs_test, dim=1)

                    # Calculate accuracy
                    accuracy_train = (
                        (predicted_labels_train == y_train).float().mean().item()
                    )
                    accuracy_test = (
                        (predicted_labels_test == y_test).float().mean().item()
                    )

                r_in_name = f" rank {r}" if r is not None else ""

                self.results.extend(
                    [
                        {
                            "epoch": epoch,
                            "optimizer": opt_name + r_in_name,
                            "lr": lr,
                            "loss": test_loss.item(),
                            "accuracy": accuracy_test,
                            "mode": "test",
                            "rank": r,
                            "eps": eps,
                            "X_type": X_type,
                            "avg_epoch_time": 0,
                            "epoch_time": 0,
                            "batch_size": batch_size,
                            "data_seed": data_seed,
                        },
                        {
                            "epoch": epoch,
                            "optimizer": opt_name + r_in_name,
                            "lr": lr,
                            "loss": train_loss.item(),
                            "accuracy": accuracy_train,
                            "mode": "train",
                            "rank": r,
                            "eps": eps,
                            "X_type": X_type,
                            "avg_epoch_time": 0,
                            "epoch_time": 0,
                            "batch_size": batch_size,
                            "data_seed": data_seed,
                        },
                    ]
                )

            all_grads = {name: [] for name, _ in model.named_parameters()}

            for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
                optimizer.zero_grad()
                y_pred = model(batch_X)
                batch_loss = criterion(y_pred, batch_y)
                batch_loss.backward()

                for name, param in model.named_parameters():
                    if param.grad is not None:
                        all_grads[name].append(param.grad.detach().cpu().numpy())

                if r:
                    optimizer.step(epoch)
                else:
                    optimizer.step()

                epoch_loss += batch_loss.item()
                num_batches += 1

            # Save gradients
            stacked_grads = {name: np.stack(grads) for name, grads in all_grads.items()}
            grad_filename = f"{task_name}_{opt_name}_lr{lr}_epoch{epoch}"
            if r is not None:
                grad_filename += f"_rank{r}"
            if grad_save_dir:
                grad_file = os.path.join(grad_save_dir, f"{grad_filename}_stacked.npz")
            np.savez_compressed(grad_file, **stacked_grads)

            # Evaluate
            y_pred_train = model(X_train)
            y_pred_test = model(X_test)

            train_loss = criterion(y_pred_train, y_train)
            test_loss = criterion(y_pred_test, y_test)

            y_pred_probs_train = torch.softmax(y_pred_train, dim=1)
            y_pred_probs_test = torch.softmax(y_pred_test, dim=1)

            predicted_labels_train = torch.argmax(y_pred_probs_train, dim=1)
            predicted_labels_test = torch.argmax(y_pred_probs_test, dim=1)

            # Calculate accuracy
            accuracy_train = (predicted_labels_train == y_train).float().mean().item()
            accuracy_test = (predicted_labels_test == y_test).float().mean().item()

            elapsed_time = time.time() - time_start
            epoch_time = time.time() - start_epoch
            avg_epoch_time = elapsed_time / (epoch + 1)

            r_in_name = f" rank {r}" if r is not None else ""

            self.results.extend(
                [
                    {
                        "epoch": epoch + 1,
                        "optimizer": opt_name + r_in_name,
                        "lr": lr,
                        "loss": test_loss.item(),
                        "accuracy": accuracy_test,
                        "mode": "test",
                        "rank": r,
                        "eps": eps,
                        "X_type": X_type,
                        "avg_epoch_time": avg_epoch_time,
                        "epoch_time": epoch_time,
                        "batch_size": batch_size,
                    },
                    {
                        "epoch": epoch + 1,
                        "optimizer": opt_name + r_in_name,
                        "lr": lr,
                        "loss": train_loss.item(),
                        "accuracy": accuracy_train,
                        "mode": "train",
                        "rank": r,
                        "eps": eps,
                        "X_type": X_type,
                        "avg_epoch_time": avg_epoch_time,
                        "epoch_time": epoch_time,
                        "batch_size": batch_size,
                    },
                ]
            )

        model.eval()
        with torch.no_grad():
            y_pred_test = model(X_test)
            test_loss = criterion(y_pred_test, y_test).item()

        return test_loss

    def run_experiment(self):
        """Run the main experiment"""
        data_seeds = self.config.get("data.data_seeds")
        in_dims = self.config.get("data.in_dims")
        out_dims = self.config.get("data.out_dims")
        enabled_tasks = self.config.get("tasks.enabled_tasks")
        learning_rates = self.config.get("training.learning_rates")
        ranks = self.config.get("training.ranks")
        epsilons = self.config.get("training.eps")

        if not data_seeds or not in_dims or not out_dims:
            raise ValueError("data_seeds or dimensions are not defined")
        base_model = self.get_model("BinClass", in_dims[0], out_dims[0])

        for data_seed in data_seeds:
            ds = self.get_dataset(in_dims[0], out_dims[0], data_seed)
            if not enabled_tasks:
                raise ValueError

            for task_name in enabled_tasks:

                X, y = ds.create_binary_data()

                scaled_dict = {"X true": X}

                print("cond", torch.linalg.cond(X))
                for Xtype in scaled_dict.keys():

                    X_train, X_test, y_train, y_test = train_test_split(
                        scaled_dict[Xtype],
                        y,
                        test_size=self.config.get("data.test_size"),
                        random_state=42,
                    )

                    print(f"X_train shape: {X_train.shape}")
                    print(f"y_train shape: {y_train.shape}")

                    final_parameters = {}

                    criterion = self.get_loss_function(task_name)

                    for opt_name, opt_config in self.config.optimizers.items():
                        if not opt_config["enabled"]:
                            continue

                        print(f"Running optimizer: {opt_name}")

                        if not learning_rates or not epsilons:
                            raise ValueError("lrs or ranks are not defined")
                        if not ranks:
                            ranks = [None]
                            rank = None

                        for lr in learning_rates:
                            for eps in epsilons:
                                if opt_config["requires_rank"]:
                                    # if len(ranks) > 0:
                                    for rank in ranks:
                                        model = copy.deepcopy(base_model)
                                        print("rank", rank)
                                        optimizer = self.get_optimizer(
                                            opt_name,
                                            model.parameters(),
                                            lr=lr,
                                            eps=eps,
                                            max_rank=rank,
                                            task=task_name,
                                        )

                                        test_loss = self.train_model_stochastic(
                                            model=model,
                                            optimizer=optimizer,
                                            criterion=criterion,
                                            X_train=X_train,
                                            y_train=y_train,
                                            X_test=X_test,
                                            y_test=y_test,
                                            opt_name=opt_name,
                                            lr=lr,
                                            eps=eps,
                                            X_type=Xtype,
                                            r=rank,
                                            data_seed=data_seed,
                                            task_name=task_name,
                                        )
                                else:
                                    # if opt_name == "Torch_Adagrad":
                                    #     eps = 1 / eps
                                    model = copy.deepcopy(base_model)
                                    optimizer = self.get_optimizer(
                                        opt_name, model.parameters(), lr, eps
                                    )

                                    test_loss = self.train_model_stochastic(
                                        model=model,
                                        optimizer=optimizer,
                                        criterion=criterion,
                                        X_train=X_train,
                                        y_train=y_train,
                                        X_test=X_test,
                                        y_test=y_test,
                                        opt_name=opt_name,
                                        eps=eps,
                                        X_type=Xtype,
                                        lr=lr,
                                        data_seed=data_seed,
                                        task_name=task_name,
                                    )

                    # Save results
                    df = pd.DataFrame(self.results)
                    df["loss"] = df["loss"].astype(float)

                    results_dir = self.config.get("output.results_dir")
                    filename = f"{task_name}_G_matrix_dynamics_{in_dims[0]}_by_{out_dims[0]}.csv"
                    if results_dir:
                        filepath = os.path.join(results_dir, filename)
                        df.to_csv(filepath)
                    else:
                        raise ValueError("results_dir is None")

                    # Plot results
                    # self.plot_results(
                    #     df, hue="optimizer", style="eps", name=filename.replace(".csv", "")
                    # )
                    # loggs_df = self.make_loggs_df()
                    # self.plot_loggs(data=loggs_df, query_condition="method == vanilla")


def main():
    """Main function to run the experiment"""
    config = Config("config.yaml")
    runner = ExperimentRunner(config)
    runner.run_experiment()


if __name__ == "__main__":
    main()
