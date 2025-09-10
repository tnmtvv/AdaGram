import torch
from torch.optim import Optimizer
import numpy as np
import os

from typing import Optional, Dict, Any, Tuple
from abc import ABC

from src.adagram_optimizers.utils.Logger import AdaGramLogger

from line_profiler import profile


class AdaGram(Optimizer, ABC):
    """
    Abstract base class for AdaGram optimizers with pluggable update strategies (different Q, P updates).

    At the end of the step, it replaces p.grad with the preconditioned gradient
    for analysis purposes.
    """

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
        enable_logging: bool = False,
        save_matrix: bool = False,
    ):

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
    
    def _faster_svd(self, state):
        ## faster svd variant with the QR  
        Qp, Rp = torch.linalg.qr(state["P"], mode='reduced')
        Qq, Rq = torch.linalg.qr(state["Q"], mode='reduced')
        small_matrix = Rp @ Rq.T
        U_s, S, Vh_s = torch.linalg.svd(small_matrix, full_matrices=False)

        if self.max_rank is not None:
            U_s = U_s[:, :self.max_rank]
            S = S[:self.max_rank]
            Vh_s = Vh_s[:self.max_rank, :]

        U = Qp @ U_s        # [n, k]
        V = Qq @ Vh_s.T     # [n, k]
        state["U"], state["S"], state["V"] = U, S, V

        state["S"] = torch.diag(state["S"])
        state["U"] = state["U"]
        state["V"] = state["V"]

    def initialize(self, state: Dict[str, Any], n: int, grad: torch.Tensor):
        """Initialize optimizer state"""
        max_rank = self.max_rank
        if not max_rank:
            max_rank = n
        
        state["L_0"] = (np.sqrt(self.eps))

        if self.enable_logging:                  
            state["G"] = self.eps * torch.eye(n, device=grad.device, dtype=grad.dtype)
            state["L_t"] = state["L_0"] * torch.eye(n, device=grad.device, dtype=grad.dtype)

            result = state["L_t"] @ state["L_t"].T
            target = state["G"]
            error_norm = torch.norm(torch.abs(target - result)) / torch.norm(target)
            print('initial norm', error_norm)

        
        state["L_0_inv"] = 1 / state["L_0"]
        state["step_count"] = 0

    def update_grad_vector(self, state, grad_vector):
        d = state["L_0_inv"]  # 1D tensor, diagonal entries
        u = grad_vector * d   # elementwise

        if "P" in state:
            P, Q = state["P"], state["Q"]
            if len(u.shape) == 1:
                u = u.view(-1, 1)
            x = Q.T @ u   # shape: (q_dim,)
            if len(x.shape) > 0:
                Px = P @ x    # shape: (n_dim,)
            else: 
                Px = P * x
            new_g_bar = (u - Px).reshape(-1)
            return new_g_bar
        else:
            new_g_bar = u.reshape(-1)
            return new_g_bar

    def update_grad_vector_sym(self, state, grad_vector):
        d = state["L_0_inv"]  # 1D tensor, diagonal entries
        u = grad_vector * d   # elementwise

        if "P" in state:
            P, Q = state["P"], state["Q"]
            if len(u.shape) == 1:
                u = u.view(-1, 1)
            x = Q.T @ u   # shape: (q_dim,)
            if len(x.shape) > 0:
                Px = P @ x    # shape: (n_dim,)
            else: 
                Px = P * x
            new_g_bar = (u - Px).reshape(-1)
        else:
            new_g_bar = u.reshape(-1)

        
        if "P" in state:
            P, Q = state["P"], state["Q"]
            if len(new_g_bar.shape) == 1:
                new_g_bar = new_g_bar.view(1, -1)
            # print(Q.shape)
            # print(new_g_bar.shape)
            x_sym = new_g_bar @ Q

            if len(x_sym.shape) > 0:
                x_sym_P = x_sym @ P.T    # shape: (n_dim,)
            else: 
                x_sym_P = x_sym * P
            new_sym_g_bar = d * (new_g_bar - x_sym_P).reshape(-1)
        else:
            new_sym_g_bar = d * new_g_bar
        return new_sym_g_bar


    def calculate_coeffs(
        self, g_bar: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculate alpha, beta coefficients"""
        g_bar_norm_sq = torch.dot(g_bar, g_bar)
        alpha = self._compute_alpha(g_bar_norm_sq)
        beta = self._compute_beta(alpha, g_bar_norm_sq)
        return g_bar_norm_sq, alpha, beta

    # @abstractmethod
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

    @profile
    def step(self, epoch: Optional[int] = None, closure=None):
        """Performs a single optimization step"""
        loss = None
        if closure is not None:
            loss = closure()

        with torch.no_grad():
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
                    
                    # Initialize state if needed
    
                    if len(state) == 0:
                        self.initialize(state, n, grad)
                        if self.enable_logging and self.save_matrix:
                            if (
                                param_idx == 0
                            ):  # Save only for first parameter to avoid too many files
                                filename = f"G_matrix_epoch_0_adagram_task_{getattr(self, 'task_name', 'unknown')}.pt"
                                torch.save(
                                    state["G"], os.path.join(self.save_dir, filename)
                                )
    
                    # Update gradient vector
                    g_bar = self.update_grad_vector(state, grad_vector)

                    ##### HERE SYM VERSION!!!
                        
                    g_bar_sym = self.update_grad_vector_sym(state, grad_vector)
    
                    g_bar_norm_sq, alpha, beta = self.calculate_coeffs(g_bar)
    
                    # Update P and Q matrices (implemented by subclasses)
                    state["P"], state["Q"], reconstruct_error = self.update_PQ(
                        state,
                        beta,
                        g_bar,
                    )
    
                    if self.enable_logging:
                        identity = torch.eye(n, device=grad.device, dtype=grad.dtype)

                        state["L_t"] = state["L_t"] @ (
                            identity + alpha * torch.ger(g_bar, g_bar)
                        )

                        state["G"] += torch.ger(grad_vector, grad_vector)
                        
                        if self.save_matrix:
                            if (
                                epoch is not None and param_idx == 0
                            ):  
                                filename = f"G_matrix_epoch_{epoch+1}_batch_{state['step_count']}_adagram_task_{getattr(self, 'task_name', 'unknown')}.pt"
                                torch.save(state["G"], os.path.join(self.save_dir, filename))
    
                        eigenvals, eigenvecs = torch.linalg.eigh(state["G"])
                        sqrt_eigenvals = torch.sqrt(eigenvals)
    
                        sqr_G = eigenvecs @ torch.diag(sqrt_eigenvals) @ eigenvecs.T
    
                        v = torch.randn(n, device=grad.device, dtype=grad.dtype)
                        v = v / torch.norm(v)
    
                        y_1 = sqr_G @ v
                        y_2 = state["L_t"] @ v
    
                        error_norm_sqr = torch.norm(y_1 - y_2) / torch.norm(y_1)
    
                        result = state["L_t"] @ state["L_t"].T
                        target = state["G"]
                        error_norm = torch.norm(torch.abs(target - result)) / torch.norm(target)

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
    
    
                    # precond_grad = g_bar / torch.sqrt(1 + g_bar_norm_sq)
                    precond_grad = g_bar_sym / (1 + g_bar_norm_sq)
                    param_vector.add_(precond_grad, alpha=-group["lr"])
                    p.grad.data = precond_grad
                    p.data = param_vector.reshape(original_shape)

        return loss
