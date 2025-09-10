import os
import random
import time
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from typing import Optional

# Assume other necessary imports (Config, custom models, optimizers, datasets) are available
# from src.adagram_optimizers.utils.Config import Config
# from src.adagram_optimizers.utils.models import ...

class Trainer:
    """
    Handles the execution of a single model training loop, including metric calculation,
    logging, and various stopping conditions based on the specified mode.
    """
    def __init__(self, config, device, results_list: list, mode: str = 'epochs'):
        """
        Initializes the Trainer.

        Args:
            config: The configuration object for the experiment.
            device: The torch device (CPU or CUDA) to run training on.
            results_list (list): A reference to the list where results should be stored.
            mode (str): The training strategy. One of ['epochs', 'time', 'plateau'].
        """
        self.config = config
        self.device = device
        self.results = results_list
        self.mode = mode
        if self.mode not in ['epochs', 'time', 'plateau']:
            raise ValueError(f"Invalid training mode specified: {self.mode}")

    def train(self, model, optimizer, criterion, X_train, y_train, X_test, y_test, **kwargs):
        """
        Public method to start the training process based on the initialized mode.
        It prepares the stop condition and delegates to the main training loop.
        """
        if self.mode == 'time':
            time_budget = self.config.get("training.time_budget", 60)
            print(f"--- Training with time budget: {time_budget} seconds ---")
            stop_condition = lambda epoch, start_time: (time.time() - start_time) < time_budget
            self._execute_training_loop(model, optimizer, criterion, X_train, y_train, X_test, y_test, stop_condition, **kwargs)

        elif self.mode == 'plateau':
            patience = self.config.get("training.patience", 10)
            val_split_ratio = self.config.get("training.early_stopping.validation_split_ratio", 0.15)
            max_epochs = self.config.get("training.num_epochs", 100)
            data_seed = kwargs.get('data_seed')

            X_train_sub, X_val, y_train_sub, y_val = train_test_split(
                X_train, y_train, test_size=val_split_ratio,
                random_state=data_seed if data_seed is not None else 42,
                stratify=y_train.cpu() if self.config.get("dataset.sparse_config.if_class") else None
            )
            print(f"Data split: {len(X_train_sub)} train, {len(X_val)} val. Early stopping with patience={patience}.")
            stop_condition = lambda epoch, start_time: epoch < max_epochs
            self._execute_training_loop(model, optimizer, criterion, X_train_sub, y_train_sub, X_test, y_test,
                                        stop_condition, X_val=X_val, y_val=y_val, patience=patience, **kwargs)

        else:  # Default to 'epochs'
            num_epochs = self.config.get("training.num_epochs")
            if not num_epochs:
                raise ValueError("num_epochs must be defined for 'epochs' training mode.")
            print(f"--- Training for a fixed {num_epochs} epochs ---")
            stop_condition = lambda epoch, start_time: epoch < num_epochs
            self._execute_training_loop(model, optimizer, criterion, X_train, y_train, X_test, y_test, stop_condition, **kwargs)

    def _execute_training_loop(self, model, optimizer, criterion, X_train, y_train, X_test, y_test,
                               stop_condition, X_val=None, y_val=None, patience=None, **kwargs):
        """The core training loop logic, generalized for any stopping condition."""
        if_class = self.config.get("dataset.sparse_config.if_class")
        train_loader = DataLoader(TensorDataset(X_train, y_train),
                                  batch_size=kwargs.get('batch_size'),
                                  shuffle=self.config.get("training.shuffle"))

        start_time = time.time()
        epoch, patience_counter, best_loss = 0, 0, float('inf')

        # Log initial metrics before training starts
        self._log_epoch_results(-1, 0, model, criterion, X_train, y_train, X_test, y_test, if_class, X_val, y_val, **kwargs)

        while stop_condition(epoch, start_time):
            model.train()
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                loss = criterion(model(batch_X), batch_y)
                loss.backward()
                optimizer.step()

            elapsed_time = time.time() - start_time
            val_metrics = self._calculate_metrics(model, criterion, X_val, y_val, if_class) if X_val is not None else (None, None, None)
            
            # Log metrics for the completed epoch
            self._log_epoch_results(epoch, elapsed_time, model, criterion, X_train, y_train, X_test, y_test, if_class, X_val, y_val, **kwargs)
            
            # Early stopping check
            if patience and X_val is not None:
                current_val_loss = val_metrics[0]
                if current_val_loss < best_loss - 1e-5:
                    best_loss, patience_counter = current_val_loss, 0
                else:
                    patience_counter += 1
                if patience_counter >= patience:
                    print(f"--- Stopping early at epoch {epoch+1} due to validation loss plateau. ---")
                    break
            epoch += 1
        model.eval()

    def _log_epoch_results(self, epoch, elapsed_time, model, criterion, X_train, y_train, X_test, y_test, if_class, X_val, y_val, **kwargs):
        """Calculates and logs metrics for a single epoch."""
        train_metrics = self._calculate_metrics(model, criterion, X_train, y_train, if_class)
        test_metrics = self._calculate_metrics(model, criterion, X_test, y_test, if_class)
        val_metrics = self._calculate_metrics(model, criterion, X_val, y_val, if_class) if X_val is not None else (None, None, None)
        
        all_metrics = (*train_metrics, *val_metrics, *test_metrics)
        self._log_results(epoch, elapsed_time, all_metrics, **kwargs)

    def _calculate_metrics(self, model, criterion, X, y, if_class):
        """Calculates loss and accuracy/RMSE for a given dataset."""
        if X is None or y is None: return None, None, None
        with torch.no_grad():
            y_pred = model(X)
            loss = criterion(y_pred, y).item()
            acc = None
            rmse = None
            if if_class:
                preds = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
                acc = (preds == y).float().mean().item()
            else:
                mse = torch.mean((y_pred.squeeze() - y.float().squeeze()) ** 2)
                rmse = torch.sqrt(mse).item()
        return loss, acc, rmse

    def _log_results(self, epoch, elapsed_time, metrics, **kwargs):
        """Appends the metrics for the current epoch to the main results list."""
        train_loss, train_acc, train_rmse, val_loss, val_acc, val_rmse, test_loss, test_acc, test_rmse = metrics
        opt_name, r = kwargs.get('opt_name'), kwargs.get('r')
        r_in_name = f" rank {r}" if r is not None and 'rank' not in opt_name else ""
        
        common_info = {
            "optimizer": opt_name + r_in_name, "epoch": epoch + 1, "epoch_time": elapsed_time,
            "avg_epoch_time": elapsed_time / (epoch + 1) if epoch > -1 else 0, **kwargs
        }

        self.results.extend([
            {**common_info, "loss": test_loss, "accuracy": test_acc, "rmse": test_rmse, "mode": "test"},
            {**common_info, "loss": train_loss, "accuracy": train_acc, "rmse": train_rmse, "mode": "train"},
        ])
        if val_loss is not None:
            self.results.append({**common_info, "loss": val_loss, "accuracy": val_acc, "rmse": val_rmse, "mode": "validation"})
