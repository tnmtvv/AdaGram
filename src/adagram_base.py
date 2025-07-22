import torch
from torch.optim import Optimizer
import numpy as np
import os

from typing import Optional, Dict, Any, Tuple
from abc import ABC, abstractmethod

from src.utils.Logger import AdaGramLogger


class AdaGram(Optimizer, ABC):
    """Abstract base class for AdaGram optimizers with pluggable update strategies"""

    def __init__(
        self,
        params,
        lr: float = 1.0,
        eps: float = 1e-10,
        weight_decay: float = 0,
        max_rank: Optional[int] = None,
        log_file: str = "results/adagram_logs.csv",
        task: str = "LinReg",
        save_dir: str = "matrix_G",
        logger: Optional[AdaGramLogger] = None,
        enable_logging: bool = True,
        save_matrix: bool = False,
    ):
        """
        Initialize AdaGram base optimizer

        Args:
            params: Model parameters
            lr: Learning rate
            eps: Epsilon for numerical stability
            weight_decay: Weight decay factor
            max_rank: Maximum rank for approximation
            task: Task name
            logger: Custom logger instance
            enable_logging: Whether to enable logging
        """
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps <= 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, eps=eps, weight_decay=weight_decay, max_rank=max_rank)
        super(AdaGram, self).__init__(params, defaults)
        self.lr = lr
        self.eps = eps
        self.max_rank = max_rank
        self.task = task
        self.enable_logging = enable_logging
        self.save_matrix = save_matrix

        # Initialize logger
        if enable_logging:
            if logger is None:
                self.logger = AdaGramLogger(
                    log_file=log_file,
                    task=task,
                    lr=lr,
                    max_rank=max_rank,
                )
            else:
                self.logger = logger
        else:
            self.logger = None

    def _compute_alpha(
        self, g_bar_norm_sq: torch.Tensor, eps: float = 1e-10
    ) -> torch.Tensor:
        """Compute alpha_t that satisfies the equation (6) in the theorem."""
        return ((1 + g_bar_norm_sq).sqrt() - 1) / g_bar_norm_sq

    def _compute_beta(
        self, alpha: torch.Tensor, g_bar_norm_sq: torch.Tensor
    ) -> torch.Tensor:
        """Compute beta_t as defined in the theorem."""
        return alpha / (1 + alpha * g_bar_norm_sq)

    def initialize(self, state: Dict[str, Any], n: int, grad: torch.Tensor):
        """Initialize optimizer state"""
        max_rank = self.max_rank
        if not max_rank:
            max_rank = n
        state["G"] = self.eps * torch.eye(n, device=grad.device, dtype=grad.dtype)

        chol_G = torch.linalg.cholesky(state["G"])

        state["L_0"] = (np.sqrt(self.eps)) * torch.eye(
            n, device=grad.device, dtype=grad.dtype
        )

        state["L_t"] = state["L_0"]
        state["L_0_inv"] = torch.linalg.inv(state["L_0"])
        print("l_0_inv: \n", state["L_0_inv"])
        state["step_count"] = 0

    def update_grad_vector(
        self, state: Dict[str, Any], grad_vector: torch.Tensor
    ) -> torch.Tensor:
        """Update gradient vector with preconditioning"""
        if "P" not in state:
            g_bar = state["L_0_inv"] @ grad_vector
        else:
            identity = torch.eye(
                grad_vector.shape[0], device=grad_vector.device, dtype=grad_vector.dtype
            )
            g_bar = (
                (identity - state["P"] @ state["Q"].T)
                @ state["L_0_inv"]
                @ grad_vector
                # torch.linalg.inv(state["L_t"])
                # @ grad_vector
            )
        return g_bar

    def calculate_coeffs(
        self, g_bar: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculate alpha, beta coefficients"""
        g_bar_norm_sq = torch.dot(g_bar, g_bar)
        alpha = self._compute_alpha(g_bar_norm_sq)
        beta = self._compute_beta(alpha, g_bar_norm_sq)
        return g_bar_norm_sq, alpha, beta

    @abstractmethod
    def update_PQ(
        self,
        state: Dict[str, Any],
        beta: torch.Tensor,
        g_bar: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Abstract method to update P and Q matrices

        Args:
            state: Optimizer state dictionary
            beta: Beta coefficient
            g_bar: Preconditioned gradient
            grad_vector: Original gradient vector
            alpha: Alpha coefficient

        Returns:
            Tuple of (P, Q, reconstruction_error)
        """
        pass

    def step(self, epoch: Optional[int] = None, closure=None):
        """Performs a single optimization step"""
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for param_idx, p in enumerate(group["params"]):
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]
                original_shape = p.data.shape

                grad_vector = grad.reshape(-1)
                param_vector = p.data.reshape(-1)
                n = len(grad_vector)
                if torch.isnan(grad_vector).any():
                    print("grad_vector", torch.linalg.norm(grad_vector))

                identity = torch.eye(n, device=grad.device, dtype=grad.dtype)

                # Initialize state if needed

                if len(state) == 0:
                    self.initialize(state, n, grad)
                    if self.save_matrix:
                        if (
                            param_idx == 0
                        ):  # Save only for first parameter to avoid too many files
                            filename = f"G_matrix_epoch_0_adagram_task_{getattr(self, 'task_name', 'unknown')}.pt"
                            torch.save(
                                state["G"], os.path.join(self.save_dir, filename)
                            )

                # Update gradient vector
                g_bar = self.update_grad_vector(state, grad_vector)

                # Update G matrix
                state["G"] += torch.ger(grad_vector, grad_vector)

                if self.save_matrix:
                    if (
                        epoch is not None and param_idx == 0
                    ):  # Save only for first parameter to avoid too many files
                        filename = f"G_matrix_epoch_{epoch+1}_batch_{state['step_count']}_adagram_task_{getattr(self, 'task_name', 'unknown')}.pt"
                        torch.save(state["G"], os.path.join(self.save_dir, filename))

                # Calculate coefficients
                g_bar_norm_sq, alpha, beta = self.calculate_coeffs(g_bar)

                # Update P and Q matrices (implemented by subclasses)
                state["P"], state["Q"], reconstruct_error = self.update_PQ(
                    state,
                    beta,
                    g_bar,
                )
                state["L_t"] = state["L_t"] @ (
                    identity + alpha * torch.ger(g_bar, g_bar)
                )

                eigenvals, eigenvecs = torch.linalg.eigh(state["G"])
                sqrt_eigenvals = torch.sqrt(eigenvals)

                sqr_G = eigenvecs @ torch.diag(sqrt_eigenvals) @ eigenvecs.T

                v = torch.randn(n)
                v = v / torch.norm(v)

                y_1 = sqr_G @ v
                y_2 = state["L_t"] @ v

                error_norm_sqr = torch.norm(y_1 - y_2) / torch.norm(y_1)

                result = state["L_t"] @ state["L_t"].T
                target = state["G"]
                error_norm = torch.norm(torch.abs(target - result)) / torch.norm(target)

                # Increment step counter
                state["step_count"] += 1

                # Log statistics
                if self.enable_logging and self.logger:
                    self.logger.log_optimizer_step(
                        step_count=state["step_count"],
                        param_id=param_idx,
                        grad_vector=grad_vector,
                        beta=beta,
                        lr=group["lr"],
                        error_norm=error_norm,
                        error_norm_sqrt=error_norm_sqr,
                        reconstruct_error=reconstruct_error,
                        state=state,
                        epoch=epoch,
                    )

                precond_grad = g_bar / torch.sqrt(1 + g_bar_norm_sq)
                param_vector.add_(precond_grad, alpha=-group["lr"])
                p.data = param_vector.reshape(original_shape)

        return loss
