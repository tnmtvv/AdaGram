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
import re

from src.adagram_fixed_rank import AdaGramFR
from src.adagram_vanilla import AdaGramVanilla
from src.adagram_projector_splitting import AdaGramPS
from src.shampoo import Shampoo
from src.full_G import FullAdaGrad
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
        max_rank: int = None,
        task: str = "LinReg",
    ):
        """Create optimizer based on configuration"""
        optimizer_map = {
            "AdaGramPS": lambda: AdaGramPS(
                params=params, lr=lr, max_rank=max_rank, task=task
            ),
            "AdaGramFR_svd": lambda: AdaGramFR(
                params, lr=lr, max_rank=max_rank, task=task
            ),
            "AdaGramFR_nosvd": lambda: AdaGramFR(params, lr=lr, max_rank=max_rank),
            "Torch_Adagrad": lambda: torch.optim.Adagrad(params, lr=lr, eps=1e-5),
            "Shampoo": lambda: Shampoo(params, lr=lr),
            "FullAdaGrad": lambda: FullAdaGrad(params=params, lr=lr, eps=1e-5),
            "AdaGram": lambda: AdaGramVanilla(params, lr=lr, eps=1e-4),
            "Vanilla_SGD": lambda: torch.optim.SGD(params, lr=lr),
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

                r_in_name = f" rank {r}" if r is not None else ""

                self.results.extend(
                    [
                        {
                            "epoch": epoch,
                            "optimizer": opt_name + r_in_name,
                            "lr": lr,
                            "loss": test_loss.item(),
                            "mode": "test",
                            "rank": r,
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
                            "mode": "train",
                            "rank": r,
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
                        "mode": "test",
                        "rank": r,
                        "avg_epoch_time": avg_epoch_time,
                        "epoch_time": epoch_time,
                        "batch_size": batch_size,
                    },
                    {
                        "epoch": epoch + 1,
                        "optimizer": opt_name + r_in_name,
                        "lr": lr,
                        "loss": train_loss.item(),
                        "mode": "train",
                        "rank": r,
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

    def extract_metadata(self, filename):
        # Remove directory and extension
        base = os.path.basename(filename)
        name, _ = os.path.splitext(base)
        # Match optimizer and rank (e.g., nosvd_adagram_logs_2)
        match = re.match(r"([a-zA-Z]+)_adagram_logs_(\d+)", name)
        if match:
            optimizer = match.group(1)
            rank = int(match.group(2))
        elif name == "adagram_vanila":
            optimizer = "vanilla"
            rank = None
        else:
            print(name)
            optimizer = "unknown"
            rank = None
        return optimizer, rank

    def make_loggs_df(self):
        csv_files = glob.glob("results/loggs/*.csv")
        print(csv_files)

        dfs = []
        for file in csv_files:
            df = pd.read_csv(file)
            optimizer, rank = self.extract_metadata(file)
            df["method"] = optimizer
            df["rank"] = rank
            dfs.append(df)

        df_logs = pd.concat(dfs, ignore_index=True)
        return df_logs

    def plot_results(self, df, name, x="epoch", y="loss", mode="test"):
        """Plot experiment results"""
        plotting_config = self.config.plotting

        plt.figure(figsize=plotting_config["figure_size"])
        grid = sns.FacetGrid(
            data=df.query(f"mode == '{mode}'"),
            col="lr",
            height=plotting_config["height"],
            aspect=plotting_config["aspect"],
            sharey=True,
        )

        grid.map_dataframe(
            sns.lineplot,
            x=x,
            y=y,
            style="optimizer",
            hue="optimizer",
            palette=plotting_config["palette"],
            linewidth=plotting_config["linewidth"],
        )

        grid.add_legend()

        in_dims = self.config.get("data.in_dims")
        out_dims = self.config.get("data.out_dims")

        title = "BinClassification"
        if in_dims and out_dims:
            title = f"BinClassification({in_dims[0]}, {out_dims[0]})"

        grid.fig.suptitle(title, fontsize=plotting_config["fontsize"]["title"])

        for ax in grid.axes.flat:
            ax.set_yscale("log")
            ax.set_xlabel(
                ax.get_xlabel(), fontsize=plotting_config["fontsize"]["axis_label"]
            )
            ax.set_ylabel(
                ax.get_ylabel(), fontsize=plotting_config["fontsize"]["axis_label"]
            )

        grid.set_titles(
            col_template="lr = {col_name}",
            fontsize=plotting_config["fontsize"]["legend_title"],
        )

        plt.tight_layout()
        plots_dir = self.config.get("output.plots_dir")
        if plots_dir:
            plt.savefig(os.path.join(plots_dir, f"{name}.pdf"))
        plt.show()

    def plot_loggs(
        self,
        data,
        query_condition=None,
        col_var="param_id",
        x_var="step",
        y_var="error_norm",
        figsize=(15, 8),
        height=5,
        aspect=1.2,
        alpha=0.7,
        linewidth=1.5,
        palette="pastel",
        legend_title="method",
        legend_title_fontsize=15,
        legend_fontsize=12,
        axis_label_fontsize=16,
        title_fontsize=20,
        col_template="param_id = {col_name}",
        suptitle=None,
        suptitle_fontsize=24,
        suptitle_y=1.02,
        log_scale_y=True,
        sharey=True,
    ):
        """
        Create a faceted line plot using seaborn FacetGrid.

        Parameters:
        -----------
        data : pd.DataFrame
            The input dataframe
        query_condition : str, optional
            Query condition to filter data (e.g., "method == 'psi' and rank == 2")
        col_var : str, default 'param_id'
            Column variable for faceting
        x_var : str, default 'step'
            Variable for x-axis
        y_var : str, default 'error_norm'
            Variable for y-axis
        figsize : tuple, default (15, 8)
            Figure size
        height : int, default 5
            Height of each facet
        aspect : float, default 1.2
            Aspect ratio of each facet
        alpha : float, default 0.7
            Line transparency
        linewidth : float, default 1.5
            Line width
        palette : str, default 'pastel'
            Color palette
        legend_title : str, default 'method'
            Title for legend
        legend_title_fontsize : int, default 15
            Font size for legend title
        legend_fontsize : int, default 12
            Font size for legend text
        axis_label_fontsize : int, default 16
            Font size for axis labels
        title_fontsize : int, default 20
            Font size for subplot titles
        col_template : str, default "param_id = {col_name}"
            Template for column titles
        suptitle : str, optional
            Super title for the entire figure
        suptitle_fontsize : int, default 24
            Font size for super title
        suptitle_y : float, default 1.02
            Y position for super title
        log_scale_y : bool, default True
            Whether to use log scale for y-axis
        sharey : bool, default True
            Whether to share y-axis across subplots

        Returns:
        --------
        grid : sns.FacetGrid
            The FacetGrid object
        """

        # Filter data if query condition is provided
        if query_condition:
            plot_data = data.query(query_condition)
        else:
            plot_data = data

        # Create figure
        plt.figure(figsize=figsize)

        # Create FacetGrid
        grid = sns.FacetGrid(
            data=plot_data, col=col_var, height=height, aspect=aspect, sharey=sharey
        )

        # Map the plotting function
        grid.map_dataframe(
            sns.lineplot,
            x=x_var,
            y=y_var,
            palette=palette,
            alpha=alpha,
            linewidth=linewidth,
        )

        # Add legend
        grid.add_legend(
            title=legend_title,
            title_fontsize=legend_title_fontsize,
            fontsize=legend_fontsize,
        )

        # Set super title if provided
        if suptitle:
            grid.fig.suptitle(suptitle, fontsize=suptitle_fontsize, y=suptitle_y)

        # Customize axes
        for ax in grid.axes.flat:
            if log_scale_y:
                ax.set_yscale("log")
            ax.set_xlabel(ax.get_xlabel(), fontsize=axis_label_fontsize)
            ax.set_ylabel(ax.get_ylabel(), fontsize=axis_label_fontsize)

        # Set column titles
        grid.set_titles(col_template=col_template, fontsize=title_fontsize)

        # Apply tight layout and show
        plt.tight_layout()
        plt.show()

        return grid

    def run_experiment(self):
        """Run the main experiment"""
        data_seeds = self.config.get("data.data_seeds")
        in_dims = self.config.get("data.in_dims")
        out_dims = self.config.get("data.out_dims")
        enabled_tasks = self.config.get("tasks.enabled_tasks")
        learning_rates = self.config.get("training.learning_rates")
        ranks = self.config.get("training.ranks")

        if not data_seeds or not in_dims or not out_dims:
            raise ValueError("data_seeds or dimensions are not defined")

        for data_seed in data_seeds:
            ds = self.get_dataset(in_dims[0], out_dims[0], data_seed)
            X, y = ds.create_data()

            print("cond", torch.linalg.cond(X))

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.config.get("data.test_size"), random_state=42
            )

            print(f"X_train shape: {X_train.shape}")
            print(f"y_train shape: {y_train.shape}")

            final_parameters = {}

            if not enabled_tasks:
                raise ValueError

            for task_name in enabled_tasks:
                criterion = self.get_loss_function(task_name)

                for opt_name, opt_config in self.config.optimizers.items():
                    if not opt_config["enabled"]:
                        continue

                    print(f"Running optimizer: {opt_name}")

                    if not learning_rates or not ranks:
                        raise ValueError("lrs or ranks are not defined")

                    for lr in learning_rates:
                        if opt_config["requires_rank"]:
                            for rank in ranks:
                                model = self.get_model(
                                    task_name, in_dims[0], out_dims[0]
                                )
                                optimizer = self.get_optimizer(
                                    opt_name, model.parameters(), lr, rank, task_name
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
                                    r=rank,
                                    data_seed=data_seed,
                                    task_name=task_name,
                                )

                                print(
                                    "weight",
                                    model.state_dict()["linear.weight"].detach(),
                                )

                                final_parameters[f"{opt_name}_rank_{rank}_lr_{lr}"] = {
                                    "weights": model.state_dict()["linear.weight"]
                                    .clone()
                                    .detach(),
                                    "bias": model.state_dict()["linear.bias"]
                                    .clone()
                                    .detach(),
                                    "final_loss": test_loss,
                                }
                        else:
                            model = self.get_model(task_name, in_dims[0], out_dims[0])
                            optimizer = self.get_optimizer(
                                opt_name, model.parameters(), lr
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
                                data_seed=data_seed,
                                task_name=task_name,
                            )

                # Save results
                df = pd.DataFrame(self.results)
                df["loss"] = df["loss"].astype(float)

                results_dir = self.config.get("output.results_dir")
                filename = f"{task_name}_ranks_all_diff_tracking_{in_dims[0]}_by_{out_dims[0]}.csv"
                if results_dir:
                    filepath = os.path.join(results_dir, filename)
                    df.to_csv(filepath)
                else:
                    raise ValueError("results_dir is None")

                # Plot results
                # self.plot_results(df, name=filename.replace(".csv", ""))
                # loggs_df = self.make_loggs_df()
                # self.plot_loggs(data=loggs_df, query_condition="method == vanilla")


def main():
    """Main function to run the experiment"""
    config = Config("config.yaml")
    runner = ExperimentRunner(config)
    runner.run_experiment()


if __name__ == "__main__":
    main()
