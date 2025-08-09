import csv
import os
import torch
import numpy as np
from typing import Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class LogEntry:
    """Data structure for a single log entry"""

    step: int
    param_id: int
    grad_norm: float
    grad_std: float
    beta: float
    lr: float
    error_norm: float
    error_norm_sqrt: float
    error_svd: float
    rank_U: int
    rank_V: int
    max_U: float
    min_U: float
    max_V: float
    min_V: float
    U_shape: tuple
    V_shape: tuple


class AdaGramLogger:
    """Dedicated logger for AdaGram optimizer statistics"""

    def __init__(
        self,
        log_file: str,
        task: str = "LinReg",
        lr: float = 1.0,
        max_rank: Optional[int] = None,
        enable_state_saving: bool = True,
    ):
        """
        Initialize the logger

        Args:
            log_file: Base path for log file
            task: Task name for identification
            lr: Learning rate for file naming
            max_rank: Maximum rank for file naming
            full_svd: Whether using full SVD
            enable_state_saving: Whether to save state matrices
        """
        self.task = task
        self.lr = lr
        self.max_rank = max_rank
        # self.full_svd = full_svd
        self.enable_state_saving = enable_state_saving

        # Generate log file path
        if log_file:
            self.log_file = log_file
        else:
            self.log_file = "results/loggs/adagram_logs.csv"

        self._initialize_csv()

        # Create state directory if needed
        if enable_state_saving:
            os.makedirs("state_G_binclass", exist_ok=True)

    def _initialize_csv(self):
        """Initialize CSV file with headers if it doesn't exist"""
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)

        if not os.path.isfile(self.log_file):
            with open(self.log_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "step",
                        "param_id",
                        "grad_norm",
                        "grad_std",
                        "beta",
                        "lr",
                        "error_norm",
                        "error_norm_sqrt",
                        "error_svd",
                        "rank_U",
                        "rank_V",
                        "max_U",
                        "min_U",
                        "max_V",
                        "min_V",
                        "U_shape_0",
                        "U_shape_1",
                        "V_shape_0",
                        "V_shape_1",
                    ]
                )

    def log_step(self, entry: LogEntry):
        """Log a single optimization step"""
        with open(self.log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    entry.step,
                    entry.param_id,
                    entry.grad_norm,
                    entry.grad_std,
                    entry.beta,
                    entry.lr,
                    entry.error_norm,
                    entry.error_norm_sqrt,
                    entry.error_svd,
                    entry.rank_U,
                    entry.rank_V,
                    entry.max_U,
                    entry.min_U,
                    entry.max_V,
                    entry.min_V,
                    entry.U_shape[0],
                    entry.U_shape[1],
                    entry.V_shape[0],
                    entry.V_shape[1],
                ]
            )

    def save_state_matrix(self, G_matrix: torch.Tensor, epoch: int):
        """Save state matrix G to compressed numpy file"""
        if not self.enable_state_saving:
            return

        G_np = G_matrix.cpu().numpy()
        filename = f"state_G_binclass/{self.task}_state_G_lr_{self.lr}_rank_{self.max_rank}_epoch_{epoch}.npz"
        np.savez_compressed(filename, G=G_np)

    def log_optimizer_step(
        self,
        step_count: int,
        param_id: int,
        grad_vector: torch.Tensor,
        beta: torch.Tensor,
        lr: float,
        error_norm: torch.Tensor,
        error_norm_sqrt: torch.Tensor,
        reconstruct_error: torch.Tensor,
        state: Dict[str, Any],
        epoch: Optional[int] = None,
    ):
        """
        Convenience method to log all optimizer statistics

        Args:
            step_count: Current step number
            param_id: Parameter index
            grad_vector: Gradient vector
            beta: Beta value
            lr: Learning rate
            error_norm: Reconstruction error norm
            reconstruct_error: SVD reconstruction error
            state: Optimizer state dictionary
            epoch: Current epoch (for state saving)
        """
        # Extract statistics
        grad_norm = torch.sqrt(torch.dot(grad_vector, grad_vector))
        grad_std = torch.std(grad_vector)

        # Extract matrix statistics
        if "P" in state and "Q" in state:
            rank_U = torch.linalg.matrix_rank(state["P"])
            rank_V = torch.linalg.matrix_rank(state["Q"])
            max_U = state["P"].max()
            max_V = state["Q"].max()
            min_U = state["P"].min()
            min_V = state["Q"].min()
            U_shape = state["P"].shape
            V_shape = state["Q"].shape
        else:
            rank_U = rank_V = max_U = max_V = min_U = min_V = torch.tensor(0)
            U_shape = V_shape = (0, 0)

        # Create log entry
        entry = LogEntry(
            step=step_count,
            param_id=param_id,
            grad_norm=grad_norm.item(),
            grad_std=grad_std.item(),
            beta=beta.item(),
            lr=lr,
            error_norm=error_norm.item(),
            error_norm_sqrt=error_norm_sqrt.item(),
            error_svd=reconstruct_error.item(),
            rank_U=rank_U.item(),
            rank_V=rank_V.item(),
            max_U=max_U.item(),
            min_U=min_U.item(),
            max_V=max_V.item(),
            min_V=min_V.item(),
            U_shape=U_shape,
            V_shape=V_shape,
        )

        # Log the entry
        self.log_step(entry)

        # Save state matrix if requested
        if epoch is not None and "G" in state:
            self.save_state_matrix(state["G"], epoch)
