from mmap import MAP_PRIVATE
import torch
from torch.optim import Optimizer
import numpy as np
import os

from typing import Optional, Dict, Any, Tuple
from abc import ABC

from adagram_optimizers import AdagramSVD
from utils.Logger import AdaGramLogger
from line_profiler import profile

from AdagramBase import AdaGram
from AdagramPS import AdaGramPS
from SymAdaGram import SymAdaGram
from AdaGram_eq import AdaGramEQ
from AdagramSVD import AdaGramFR
from AdagramSqrtSVD import AdaGramFR_Sqrt
from AdagramSqrtPS import AdaGramPS_Sqrt


class AdamGram(AdaGramPS):
    """
    AdaGram optimizer with Adam momentum integration.
    
    Combines AdaGram preconditioning with Adam's adaptive moment estimation.
    """

    def __init__(
        self,
        params,
        lr: float = 0.001,  
        beta1: float = 0.9, 
        beta2: float = 0.999,
        alpha = None,
        eps: float = 1e-8,  
        weight_decay: float = 0,
        max_rank: Optional[int] = None,
        log_file: str = "results/adagram_logs.csv",
        task: str = "LinReg",
        save_dir: str = "matrix_G",
        logger: Optional[AdaGramLogger] = None,
        enable_logging: bool = False,
        save_matrix: bool = False,
    ):
        # Validate parameters BEFORE creating defaults dict
        if not isinstance(lr, (int, float)):
            raise TypeError(f"lr must be a number, got {type(lr)}: {lr}")
        if not isinstance(eps, (int, float)):
            raise TypeError(f"eps must be a number, got {type(eps)}: {eps}")
        if not isinstance(weight_decay, (int, float)):
            raise TypeError(f"weight_decay must be a number, got {type(weight_decay)}: {weight_decay}")
        if not isinstance(beta1, (int, float)):
            raise TypeError(f"beta1 must be a number, got {type(beta1)}: {beta1}")
        if not isinstance(beta2, (int, float)):
            raise TypeError(f"beta2 must be a number, got {type(beta2)}: {beta2}")
        
        # Convert to float
        lr = float(lr)
        eps = float(eps)
        weight_decay = float(weight_decay)
        beta1 = float(beta1)
        beta2 = float(beta2)
        
        # Now validate values
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps <= 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= beta1 < 1.0:
            raise ValueError(f"Invalid beta1 parameter: {beta1}")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError(f"Invalid beta2 parameter: {beta2}")

        defaults = dict(
            lr=lr, 
            eps=eps, 
            weight_decay=weight_decay, 
            max_rank=max_rank,
            beta1=beta1,
            beta2=beta2
        )
        
        # Call Optimizer.__init__ directly to avoid MRO issues
        # This bypasses potential issues with multiple inheritance
        torch.optim.Optimizer.__init__(self, params, defaults)
        
        # Store instance variables
        self.lr = lr
        self.eps = eps
        self.beta1 = beta1
        self.beta2 = beta2
        self.max_rank = max_rank
        self.alpha = alpha
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

    def initialize(self, state, n, grad):
        """Initialize both AdaGram and Adam state variables"""
        # AdaGram initialization (call parent initialization)
        super().initialize(state, n, grad)
        
        # Adam state initialization
        state['m_t'] = torch.zeros_like(grad).reshape(-1)  # First moment
        # state['v_t'] = torch.zeros_like(grad).reshape(-1)  # Second moment

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

                    
                    # v_t = self.beta2 * v_prev + (1 - self.beta2) * precond_grad.mul(precond_grad)

                    # # Adam update
                    # adam_update = group["lr"] * m_t_hat / (v_t_hat.sqrt() + group["eps"])

                    # # Store updated Adam state
                    # state['m_t'] = m_t  # Save updated state
                    # state['v_t'] = v_t  # Save updated state

                    grad_vector = grad.reshape(-1)
                    param_vector = p.data.reshape(-1)
                    n = len(grad_vector)

                    
                    # Initialize state if needed
                    if len(state) == 0:
                        self.initialize(state, n, grad)
                        if self.enable_logging and self.save_matrix:
                            if param_idx == 0:
                                filename = f"G_matrix_epoch_0_adagram_task_{getattr(self, 'task_name', 'unknown')}.pt"
                                torch.save(
                                    state["G"], os.path.join(self.save_dir, filename)
                                )

                    # AdaGram preconditioning


                    moment_grad = self.beta1 * state["m_t"] + (1 - self.beta1) * grad_vector

                    g_bar = self.update_grad_vector(state, moment_grad)
                    g_bar_norm_sq, alpha, beta = self.calculate_coeffs(g_bar)

                    # Update P and Q matrices (AdaGram logic)
                    
                    state["P"], state["Q"], reconstruct_error = self.update_PQ(
                        state,
                        beta,
                        g_bar,
                    )

                    # Logging logic (unchanged)
                    if self.enable_logging:
                        identity = torch.eye(n, device=grad.device, dtype=grad.dtype)
                        state["L_t"] = state["L_t"] @ (
                            identity + alpha * torch.ger(g_bar, g_bar)
                        )
                        state["G"] += torch.ger(grad_vector, grad_vector)
                        
                        if self.save_matrix:
                            if epoch is not None and param_idx == 0:
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

                    state['m_t'] = moment_grad 


                    precond_grad = g_bar / torch.sqrt(1 + g_bar_norm_sq) ## v
                    
                    m_hat = state["m_t"] / (1 - self.beta1**state["step_count"])
                    v_hat = precond_grad / (1 - self.beta2**state["step_count"])

                    update = m_hat / (torch.sqrt(v_hat) + 1e-3)

                    param_vector.add_(precond_grad, alpha=-group["lr"])
                    # param_vector.add_(update, alpha=-group["lr"])
                    # Update gradient for analysis (optional)
                    p.grad.data = precond_grad.reshape(original_shape)
                    p.data = param_vector.reshape(original_shape)

        return loss


class SymAdamGram(SymAdaGram):
    """
    AdaGram optimizer with Adam momentum integration.
    
    Combines AdaGram preconditioning with Adam's adaptive moment estimation.
    """

    def __init__(
        self,
        params,
        lr: float = 0.001,  
        beta1: float = 0.9, 
        beta2: float = 0.999,
        alpha = None,
        eps: float = 1e-8,  
        weight_decay: float = 0,
        max_rank: Optional[int] = None,
        log_file: str = "results/adagram_logs.csv",
        task: str = "LinReg",
        save_dir: str = "matrix_G",
        logger: Optional[AdaGramLogger] = None,
        enable_logging: bool = False,
        save_matrix: bool = False,
    ):
        # Validate parameters BEFORE creating defaults dict
        if not isinstance(lr, (int, float)):
            raise TypeError(f"lr must be a number, got {type(lr)}: {lr}")
        if not isinstance(eps, (int, float)):
            raise TypeError(f"eps must be a number, got {type(eps)}: {eps}")
        if not isinstance(weight_decay, (int, float)):
            raise TypeError(f"weight_decay must be a number, got {type(weight_decay)}: {weight_decay}")
        if not isinstance(beta1, (int, float)):
            raise TypeError(f"beta1 must be a number, got {type(beta1)}: {beta1}")
        if not isinstance(beta2, (int, float)):
            raise TypeError(f"beta2 must be a number, got {type(beta2)}: {beta2}")
        
        # Convert to float
        lr = float(lr)
        eps = float(eps)
        weight_decay = float(weight_decay)
        beta1 = float(beta1)
        beta2 = float(beta2)
        
        # Now validate values
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps <= 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= beta1 < 1.0:
            raise ValueError(f"Invalid beta1 parameter: {beta1}")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError(f"Invalid beta2 parameter: {beta2}")

        defaults = dict(
            lr=lr, 
            eps=eps, 
            weight_decay=weight_decay, 
            max_rank=max_rank,
            beta1=beta1,
            beta2=beta2
        )
        
        # Call Optimizer.__init__ directly to avoid MRO issues
        # This bypasses potential issues with multiple inheritance
        torch.optim.Optimizer.__init__(self, params, defaults)
        
        # Store instance variables
        self.lr = lr
        self.eps = eps
        self.beta1 = beta1
        self.beta2 = beta2
        self.max_rank = max_rank
        self.alpha = alpha
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

    def initialize(self, state, n, grad):
        """Initialize both AdaGram and Adam state variables"""
        # AdaGram initialization (call parent initialization)
        super().initialize(state, n, grad)
        
        # Adam state initialization
        state['m_t'] = torch.zeros_like(grad).reshape(-1)  # First moment
        # state['v_t'] = torch.zeros_like(grad).reshape(-1)  # Second moment

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

                    
                    # v_t = self.beta2 * v_prev + (1 - self.beta2) * precond_grad.mul(precond_grad)

                    # # Adam update
                    # adam_update = group["lr"] * m_t_hat / (v_t_hat.sqrt() + group["eps"])

                    # # Store updated Adam state
                    # state['m_t'] = m_t  # Save updated state
                    # state['v_t'] = v_t  # Save updated state

                    grad_vector = grad.reshape(-1)
                    param_vector = p.data.reshape(-1)
                    n = len(grad_vector)

                    
                    # Initialize state if needed
                    if len(state) == 0:
                        self.initialize(state, n, grad)
                        if self.enable_logging and self.save_matrix:
                            if param_idx == 0:
                                filename = f"G_matrix_epoch_0_adagram_task_{getattr(self, 'task_name', 'unknown')}.pt"
                                torch.save(
                                    state["G"], os.path.join(self.save_dir, filename)
                                )

                    # AdaGram preconditioning


                    moment_grad = self.beta1 * state["m_t"] + (1 - self.beta1) * grad_vector

                    g_bar = self.update_grad_vector(state, moment_grad)
                    g_bar_norm_sq, alpha, beta = self.calculate_coeffs(g_bar)

                    # Update P and Q matrices (AdaGram logic)
                    
                    state["P"], state["Q"], reconstruct_error = self.update_PQ(
                        state,
                        beta,
                        g_bar,
                    )

                    # Logging logic (unchanged)
                    if self.enable_logging:
                        identity = torch.eye(n, device=grad.device, dtype=grad.dtype)
                        state["L_t"] = state["L_t"] @ (
                            identity + alpha * torch.ger(g_bar, g_bar)
                        )
                        state["G"] += torch.ger(grad_vector, grad_vector)
                        
                        if self.save_matrix:
                            if epoch is not None and param_idx == 0:
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

                    state['m_t'] = moment_grad 


                    precond_grad = g_bar / torch.sqrt(1 + g_bar_norm_sq) ## v
                    
                    m_hat = state["m_t"] / (1 - self.beta1**state["step_count"])
                    v_hat = precond_grad / (1 - self.beta2**state["step_count"])

                    update = m_hat / (torch.sqrt(v_hat) + 1e-3)

                    param_vector.add_(precond_grad, alpha=-group["lr"])
                    # param_vector.add_(update, alpha=-group["lr"])
                    # Update gradient for analysis (optional)
                    p.grad.data = precond_grad.reshape(original_shape)
                    p.data = param_vector.reshape(original_shape)

        return loss


class SymAdamGram(SymAdaGram):
    """
    AdaGram optimizer with Adam momentum integration.
    
    Combines AdaGram preconditioning with Adam's adaptive moment estimation.
    """

    def __init__(
        self,
        params,
        lr: float = 0.001,  
        beta1: float = 0.9, 
        beta2: float = 0.999,
        alpha = None,
        eps: float = 1e-8,  
        weight_decay: float = 0,
        max_rank: Optional[int] = None,
        log_file: str = "results/adagram_logs.csv",
        task: str = "LinReg",
        save_dir: str = "matrix_G",
        logger: Optional[AdaGramLogger] = None,
        enable_logging: bool = False,
        save_matrix: bool = False,
    ):
        # Validate parameters BEFORE creating defaults dict
        if not isinstance(lr, (int, float)):
            raise TypeError(f"lr must be a number, got {type(lr)}: {lr}")
        if not isinstance(eps, (int, float)):
            raise TypeError(f"eps must be a number, got {type(eps)}: {eps}")
        if not isinstance(weight_decay, (int, float)):
            raise TypeError(f"weight_decay must be a number, got {type(weight_decay)}: {weight_decay}")
        if not isinstance(beta1, (int, float)):
            raise TypeError(f"beta1 must be a number, got {type(beta1)}: {beta1}")
        if not isinstance(beta2, (int, float)):
            raise TypeError(f"beta2 must be a number, got {type(beta2)}: {beta2}")
        
        # Convert to float
        lr = float(lr)
        eps = float(eps)
        weight_decay = float(weight_decay)
        beta1 = float(beta1)
        beta2 = float(beta2)
        
        # Now validate values
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps <= 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= beta1 < 1.0:
            raise ValueError(f"Invalid beta1 parameter: {beta1}")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError(f"Invalid beta2 parameter: {beta2}")

        defaults = dict(
            lr=lr, 
            eps=eps, 
            weight_decay=weight_decay, 
            max_rank=max_rank,
            beta1=beta1,
            beta2=beta2
        )
        
        # Call Optimizer.__init__ directly to avoid MRO issues
        # This bypasses potential issues with multiple inheritance
        torch.optim.Optimizer.__init__(self, params, defaults)
        
        # Store instance variables
        self.lr = lr
        self.eps = eps
        self.beta1 = beta1
        self.beta2 = beta2
        self.max_rank = max_rank
        self.alpha = alpha
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

    def initialize(self, state, n, grad):
        """Initialize both AdaGram and Adam state variables"""
        # AdaGram initialization (call parent initialization)
        super().initialize(state, n, grad)
        
        # Adam state initialization
        state['m_t'] = torch.zeros_like(grad).reshape(-1)  # First moment
        # state['v_t'] = torch.zeros_like(grad).reshape(-1)  # Second moment

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

                    
                    # v_t = self.beta2 * v_prev + (1 - self.beta2) * precond_grad.mul(precond_grad)

                    # # Adam update
                    # adam_update = group["lr"] * m_t_hat / (v_t_hat.sqrt() + group["eps"])

                    # # Store updated Adam state
                    # state['m_t'] = m_t  # Save updated state
                    # state['v_t'] = v_t  # Save updated state

                    grad_vector = grad.reshape(-1)
                    param_vector = p.data.reshape(-1)
                    n = len(grad_vector)

                    
                    # Initialize state if needed
                    if len(state) == 0:
                        self.initialize(state, n, grad)
                        if self.enable_logging and self.save_matrix:
                            if param_idx == 0:
                                filename = f"G_matrix_epoch_0_adagram_task_{getattr(self, 'task_name', 'unknown')}.pt"
                                torch.save(
                                    state["G"], os.path.join(self.save_dir, filename)
                                )

                    # AdaGram preconditioning


                    moment_grad = self.beta1 * state["m_t"] + (1 - self.beta1) * grad_vector

                    g_bar = self.update_grad_vector(state, moment_grad)
                    g_bar_norm_sq, alpha, beta = self.calculate_coeffs(g_bar)

                    # Update P and Q matrices (AdaGram logic)
                    
                    state["P"], state["Q"], reconstruct_error = self.update_PQ(
                        state,
                        beta,
                        g_bar,
                    )

                    # Logging logic (unchanged)
                    if self.enable_logging:
                        identity = torch.eye(n, device=grad.device, dtype=grad.dtype)
                        state["L_t"] = state["L_t"] @ (
                            identity + alpha * torch.ger(g_bar, g_bar)
                        )
                        state["G"] += torch.ger(grad_vector, grad_vector)
                        
                        if self.save_matrix:
                            if epoch is not None and param_idx == 0:
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

                    state['m_t'] = moment_grad 


                    precond_grad = g_bar / torch.sqrt(1 + g_bar_norm_sq) ## v
                    
                    m_hat = state["m_t"] / (1 - self.beta1**state["step_count"])
                    v_hat = precond_grad / (1 - self.beta2**state["step_count"])

                    update = m_hat / (torch.sqrt(v_hat) + 1e-3)

                    param_vector.add_(precond_grad, alpha=-group["lr"])
                    # param_vector.add_(update, alpha=-group["lr"])
                    # Update gradient for analysis (optional)
                    p.grad.data = precond_grad.reshape(original_shape)
                    p.data = param_vector.reshape(original_shape)

        return loss


class EQAdamGram(AdaGramEQ):
    """
    AdaGram optimizer with Adam momentum integration.
    
    Combines AdaGram preconditioning with Adam's adaptive moment estimation.
    """

    def __init__(
        self,
        params,
        lr: float = 0.001,  
        beta1: float = 0.9, 
        beta2: float = 0.999,
        alpha = None,
        eps: float = 1e-8,  
        weight_decay: float = 0,
        max_rank: Optional[int] = None,
        log_file: str = "results/adagram_logs.csv",
        task: str = "LinReg",
        save_dir: str = "matrix_G",
        logger: Optional[AdaGramLogger] = None,
        enable_logging: bool = False,
        save_matrix: bool = False,
    ):
        # Validate parameters BEFORE creating defaults dict
        if not isinstance(lr, (int, float)):
            raise TypeError(f"lr must be a number, got {type(lr)}: {lr}")
        if not isinstance(eps, (int, float)):
            raise TypeError(f"eps must be a number, got {type(eps)}: {eps}")
        if not isinstance(weight_decay, (int, float)):
            raise TypeError(f"weight_decay must be a number, got {type(weight_decay)}: {weight_decay}")
        if not isinstance(beta1, (int, float)):
            raise TypeError(f"beta1 must be a number, got {type(beta1)}: {beta1}")
        if not isinstance(beta2, (int, float)):
            raise TypeError(f"beta2 must be a number, got {type(beta2)}: {beta2}")
        
        # Convert to float
        lr = float(lr)
        eps = float(eps)
        weight_decay = float(weight_decay)
        beta1 = float(beta1)
        beta2 = float(beta2)
        
        # Now validate values
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps <= 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= beta1 < 1.0:
            raise ValueError(f"Invalid beta1 parameter: {beta1}")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError(f"Invalid beta2 parameter: {beta2}")

        defaults = dict(
            lr=lr, 
            eps=eps, 
            weight_decay=weight_decay, 
            max_rank=max_rank,
            beta1=beta1,
            beta2=beta2
        )
        
        # Call Optimizer.__init__ directly to avoid MRO issues
        # This bypasses potential issues with multiple inheritance
        torch.optim.Optimizer.__init__(self, params, defaults)
        
        # Store instance variables
        self.lr = lr
        self.eps = eps
        self.beta1 = beta1
        self.beta2 = beta2
        self.max_rank = max_rank
        self.alpha = alpha
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

    def initialize(self, state, n, grad):
        """Initialize both AdaGram and Adam state variables"""
        # AdaGram initialization (call parent initialization)
        super().initialize(state, n, grad)
        
        # Adam state initialization
        state['m_t'] = torch.zeros_like(grad).reshape(-1)  # First moment
        # state['v_t'] = torch.zeros_like(grad).reshape(-1)  # Second moment

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

                    
                    # v_t = self.beta2 * v_prev + (1 - self.beta2) * precond_grad.mul(precond_grad)

                    # # Adam update
                    # adam_update = group["lr"] * m_t_hat / (v_t_hat.sqrt() + group["eps"])

                    # # Store updated Adam state
                    # state['m_t'] = m_t  # Save updated state
                    # state['v_t'] = v_t  # Save updated state

                    grad_vector = grad.reshape(-1)
                    param_vector = p.data.reshape(-1)
                    n = len(grad_vector)

                    
                    # Initialize state if needed
                    if len(state) == 0:
                        self.initialize(state, n, grad)
                        if self.enable_logging and self.save_matrix:
                            if param_idx == 0:
                                filename = f"G_matrix_epoch_0_adagram_task_{getattr(self, 'task_name', 'unknown')}.pt"
                                torch.save(
                                    state["G"], os.path.join(self.save_dir, filename)
                                )

                    # AdaGram preconditioning


                    moment_grad = self.beta1 * state["m_t"] + (1 - self.beta1) * grad_vector

                    g_bar = self.update_grad_vector(state, moment_grad)
                    g_bar_norm_sq, alpha, beta = self.calculate_coeffs(g_bar)

                    # Update P and Q matrices (AdaGram logic)
                    
                    state["P"], state["Q"], reconstruct_error = self.update_PQ(
                        state,
                        beta,
                        g_bar,
                    )

                    # Logging logic (unchanged)
                    if self.enable_logging:
                        identity = torch.eye(n, device=grad.device, dtype=grad.dtype)
                        state["L_t"] = state["L_t"] @ (
                            identity + alpha * torch.ger(g_bar, g_bar)
                        )
                        state["G"] += torch.ger(grad_vector, grad_vector)
                        
                        if self.save_matrix:
                            if epoch is not None and param_idx == 0:
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

                    state['m_t'] = moment_grad 


                    precond_grad = g_bar / torch.sqrt(1 + g_bar_norm_sq) ## v
                    
                    m_hat = state["m_t"] / (1 - self.beta1**state["step_count"])
                    v_hat = precond_grad / (1 - self.beta2**state["step_count"])

                    update = m_hat / (torch.sqrt(v_hat) + 1e-3)

                    param_vector.add_(precond_grad, alpha=-group["lr"])
                    # param_vector.add_(update, alpha=-group["lr"])
                    # Update gradient for analysis (optional)
                    p.grad.data = precond_grad.reshape(original_shape)
                    p.data = param_vector.reshape(original_shape)

        return loss


class SVDAdamGram(AdaGramFR):
    """
    AdaGram optimizer with Adam momentum integration.
    
    Combines AdaGram preconditioning with Adam's adaptive moment estimation.
    """

    def __init__(
        self,
        params,
        lr: float = 0.001,  
        beta1: float = 0.9, 
        beta2: float = 0.999,
        alpha = None,
        eps: float = 1e-8,  
        weight_decay: float = 0,
        max_rank: Optional[int] = None,
        log_file: str = "results/adagram_logs.csv",
        task: str = "LinReg",
        save_dir: str = "matrix_G",
        logger: Optional[AdaGramLogger] = None,
        enable_logging: bool = False,
        save_matrix: bool = False,
    ):
        # Validate parameters BEFORE creating defaults dict
        if not isinstance(lr, (int, float)):
            raise TypeError(f"lr must be a number, got {type(lr)}: {lr}")
        if not isinstance(eps, (int, float)):
            raise TypeError(f"eps must be a number, got {type(eps)}: {eps}")
        if not isinstance(weight_decay, (int, float)):
            raise TypeError(f"weight_decay must be a number, got {type(weight_decay)}: {weight_decay}")
        if not isinstance(beta1, (int, float)):
            raise TypeError(f"beta1 must be a number, got {type(beta1)}: {beta1}")
        if not isinstance(beta2, (int, float)):
            raise TypeError(f"beta2 must be a number, got {type(beta2)}: {beta2}")
        
        # Convert to float
        lr = float(lr)
        eps = float(eps)
        weight_decay = float(weight_decay)
        beta1 = float(beta1)
        beta2 = float(beta2)
        
        # Now validate values
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps <= 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= beta1 < 1.0:
            raise ValueError(f"Invalid beta1 parameter: {beta1}")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError(f"Invalid beta2 parameter: {beta2}")

        defaults = dict(
            lr=lr, 
            eps=eps, 
            weight_decay=weight_decay, 
            max_rank=max_rank,
            beta1=beta1,
            beta2=beta2
        )
        
        # Call Optimizer.__init__ directly to avoid MRO issues
        # This bypasses potential issues with multiple inheritance
        torch.optim.Optimizer.__init__(self, params, defaults)
        
        # Store instance variables
        self.lr = lr
        self.eps = eps
        self.beta1 = beta1
        self.beta2 = beta2
        self.max_rank = max_rank
        self.alpha = alpha
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

    def initialize(self, state, n, grad):
        """Initialize both AdaGram and Adam state variables"""
        # AdaGram initialization (call parent initialization)
        super().initialize(state, n, grad)
        
        # Adam state initialization
        state['m_t'] = torch.zeros_like(grad).reshape(-1)  # First moment
        # state['v_t'] = torch.zeros_like(grad).reshape(-1)  # Second moment

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
                            if param_idx == 0:
                                filename = f"G_matrix_epoch_0_adagram_task_{getattr(self, 'task_name', 'unknown')}.pt"
                                torch.save(
                                    state["G"], os.path.join(self.save_dir, filename)
                                )

                    # AdaGram preconditioning


                    moment_grad = self.beta1 * state["m_t"] + (1 - self.beta1) * grad_vector

                    g_bar = self.update_grad_vector(state, moment_grad)
                    g_bar_norm_sq, alpha, beta = self.calculate_coeffs(g_bar)

                    # Update P and Q matrices (AdaGram logic)
                    
                    state["P"], state["Q"], reconstruct_error = self.update_PQ(
                        state,
                        beta,
                        g_bar,
                    )

                    # Logging logic (unchanged)
                    if self.enable_logging:
                        identity = torch.eye(n, device=grad.device, dtype=grad.dtype)
                        state["L_t"] = state["L_t"] @ (
                            identity + alpha * torch.ger(g_bar, g_bar)
                        )
                        state["G"] += torch.ger(grad_vector, grad_vector)
                        
                        if self.save_matrix:
                            if epoch is not None and param_idx == 0:
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

                    state['m_t'] = moment_grad 


                    precond_grad = g_bar / torch.sqrt(1 + g_bar_norm_sq) ## v
                    
                    m_hat = state["m_t"] / (1 - self.beta1**state["step_count"])
                    v_hat = precond_grad / (1 - self.beta2**state["step_count"])

                    update = m_hat / (torch.sqrt(v_hat) + 1e-3)

                    param_vector.add_(precond_grad, alpha=-group["lr"])
                    # param_vector.add_(update, alpha=-group["lr"])
                    # Update gradient for analysis (optional)
                    p.grad.data = precond_grad.reshape(original_shape)
                    p.data = param_vector.reshape(original_shape)

        return loss


class AdamGramSqrt_SVD(AdaGramFR_Sqrt):
    """
    AdaGram optimizer with Adam momentum integration.
    
    Combines AdaGram preconditioning with Adam's adaptive moment estimation.
    """

    def __init__(
        self,
        params,
        lr: float = 0.001,  
        beta1: float = 0.9, 
        beta2: float = 0.999,
        alpha = None,
        eps: float = 1e-8,  
        weight_decay: float = 0,
        max_rank: Optional[int] = None,
        log_file: str = "results/adagram_logs.csv",
        task: str = "LinReg",
        save_dir: str = "matrix_G",
        logger: Optional[AdaGramLogger] = None,
        enable_logging: bool = False,
        save_matrix: bool = False,
    ):
        # Validate parameters BEFORE creating defaults dict
        if not isinstance(lr, (int, float)):
            raise TypeError(f"lr must be a number, got {type(lr)}: {lr}")
        if not isinstance(eps, (int, float)):
            raise TypeError(f"eps must be a number, got {type(eps)}: {eps}")
        if not isinstance(weight_decay, (int, float)):
            raise TypeError(f"weight_decay must be a number, got {type(weight_decay)}: {weight_decay}")
        if not isinstance(beta1, (int, float)):
            raise TypeError(f"beta1 must be a number, got {type(beta1)}: {beta1}")
        if not isinstance(beta2, (int, float)):
            raise TypeError(f"beta2 must be a number, got {type(beta2)}: {beta2}")
        
        # Convert to float
        lr = float(lr)
        eps = float(eps)
        weight_decay = float(weight_decay)
        beta1 = float(beta1)
        beta2 = float(beta2)
        
        # Now validate values
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps <= 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= beta1 < 1.0:
            raise ValueError(f"Invalid beta1 parameter: {beta1}")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError(f"Invalid beta2 parameter: {beta2}")

        defaults = dict(
            lr=lr, 
            eps=eps, 
            weight_decay=weight_decay, 
            max_rank=max_rank,
            beta1=beta1,
            beta2=beta2
        )
        
        # Call Optimizer.__init__ directly to avoid MRO issues
        # This bypasses potential issues with multiple inheritance
        torch.optim.Optimizer.__init__(self, params, defaults)
        
        # Store instance variables
        self.lr = lr
        self.eps = eps
        self.beta1 = beta1
        self.beta2 = beta2
        self.max_rank = max_rank
        self.alpha = alpha
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

    def initialize(self, state, n, grad):
        """Initialize both AdaGram and Adam state variables"""
        # AdaGram initialization (call parent initialization)
        super().initialize(state, n, grad)
        
        # Adam state initialization
        state['m_t'] = torch.zeros_like(grad).reshape(-1)  # First moment
        # state['v_t'] = torch.zeros_like(grad).reshape(-1)  # Second moment

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
                            if param_idx == 0:
                                filename = f"G_matrix_epoch_0_adagram_task_{getattr(self, 'task_name', 'unknown')}.pt"
                                torch.save(
                                    state["G"], os.path.join(self.save_dir, filename)
                                )

                    # AdaGram preconditioning


                    moment_grad = self.beta1 * state["m_t"] + (1 - self.beta1) * grad_vector

                    g_bar = self.update_grad_vector(state, moment_grad)
                    g_bar_norm_sq, alpha, beta = self.calculate_coeffs(g_bar)

                    # Update P and Q matrices (AdaGram logic)
                    
                    state["P"], state["Q"], reconstruct_error = self.update_PQ(
                        state,
                        beta,
                        g_bar,
                    )

                    # Logging logic (unchanged)
                    if self.enable_logging:
                        identity = torch.eye(n, device=grad.device, dtype=grad.dtype)
                        state["L_t"] = state["L_t"] @ (
                            identity + alpha * torch.ger(g_bar, g_bar)
                        )
                        state["G"] += torch.ger(grad_vector, grad_vector)
                        
                        if self.save_matrix:
                            if epoch is not None and param_idx == 0:
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

                    state['m_t'] = moment_grad 


                    precond_grad = g_bar / torch.sqrt(1 + g_bar_norm_sq) ## v
                    
                    m_hat = state["m_t"] / (1 - self.beta1**state["step_count"])
                    v_hat = precond_grad / (1 - self.beta2**state["step_count"])

                    update = m_hat / (torch.sqrt(v_hat) + 1e-3)

                    param_vector.add_(precond_grad, alpha=-group["lr"])
                    # param_vector.add_(update, alpha=-group["lr"])
                    # Update gradient for analysis (optional)
                    p.grad.data = precond_grad.reshape(original_shape)
                    p.data = param_vector.reshape(original_shape)

        return loss



class AdamGramSqrt_PS(AdaGramPS_Sqrt):
    """
    AdaGram optimizer with Adam momentum integration.
    
    Combines AdaGram preconditioning with Adam's adaptive moment estimation.
    """

    def __init__(
        self,
        params,
        lr: float = 0.001,  
        beta1: float = 0.9, 
        beta2: float = 0.999,
        alpha = None,
        eps: float = 1e-8,  
        weight_decay: float = 0,
        max_rank: Optional[int] = None,
        log_file: str = "results/adagram_logs.csv",
        task: str = "LinReg",
        save_dir: str = "matrix_G",
        logger: Optional[AdaGramLogger] = None,
        enable_logging: bool = False,
        save_matrix: bool = False,
    ):
        # Validate parameters BEFORE creating defaults dict
        if not isinstance(lr, (int, float)):
            raise TypeError(f"lr must be a number, got {type(lr)}: {lr}")
        if not isinstance(eps, (int, float)):
            raise TypeError(f"eps must be a number, got {type(eps)}: {eps}")
        if not isinstance(weight_decay, (int, float)):
            raise TypeError(f"weight_decay must be a number, got {type(weight_decay)}: {weight_decay}")
        if not isinstance(beta1, (int, float)):
            raise TypeError(f"beta1 must be a number, got {type(beta1)}: {beta1}")
        if not isinstance(beta2, (int, float)):
            raise TypeError(f"beta2 must be a number, got {type(beta2)}: {beta2}")
        
        # Convert to float
        lr = float(lr)
        eps = float(eps)
        weight_decay = float(weight_decay)
        beta1 = float(beta1)
        beta2 = float(beta2)
        
        # Now validate values
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps <= 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= beta1 < 1.0:
            raise ValueError(f"Invalid beta1 parameter: {beta1}")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError(f"Invalid beta2 parameter: {beta2}")

        defaults = dict(
            lr=lr, 
            eps=eps, 
            weight_decay=weight_decay, 
            max_rank=max_rank,
            beta1=beta1,
            beta2=beta2
        )
        
        # Call Optimizer.__init__ directly to avoid MRO issues
        # This bypasses potential issues with multiple inheritance
        torch.optim.Optimizer.__init__(self, params, defaults)
        
        # Store instance variables
        self.lr = lr
        self.eps = eps
        self.beta1 = beta1
        self.beta2 = beta2
        self.max_rank = max_rank
        self.alpha = alpha
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

    def initialize(self, state, n, grad):
        """Initialize both AdaGram and Adam state variables"""
        # AdaGram initialization (call parent initialization)
        super().initialize(state, n, grad)
        
        # Adam state initialization
        state['m_t'] = torch.zeros_like(grad).reshape(-1)  # First moment
        # state['v_t'] = torch.zeros_like(grad).reshape(-1)  # Second moment

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
                            if param_idx == 0:
                                filename = f"G_matrix_epoch_0_adagram_task_{getattr(self, 'task_name', 'unknown')}.pt"
                                torch.save(
                                    state["G"], os.path.join(self.save_dir, filename)
                                )

                    # AdaGram preconditioning


                    moment_grad = self.beta1 * state["m_t"] + (1 - self.beta1) * grad_vector

                    g_bar = self.update_grad_vector(state, moment_grad)
                    g_bar_norm_sq, alpha, beta = self.calculate_coeffs(g_bar)

                    # Update P and Q matrices (AdaGram logic)
                    
                    state["P"], state["Q"], reconstruct_error = self.update_PQ(
                        state,
                        beta,
                        g_bar,
                    )

                    # Logging logic (unchanged)
                    if self.enable_logging:
                        identity = torch.eye(n, device=grad.device, dtype=grad.dtype)
                        state["L_t"] = state["L_t"] @ (
                            identity + alpha * torch.ger(g_bar, g_bar)
                        )
                        state["G"] += torch.ger(grad_vector, grad_vector)
                        
                        if self.save_matrix:
                            if epoch is not None and param_idx == 0:
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

                    state['m_t'] = moment_grad 


                    precond_grad = g_bar / torch.sqrt(1 + g_bar_norm_sq) ## v
                    
                    m_hat = state["m_t"] / (1 - self.beta1**state["step_count"])
                    v_hat = precond_grad / (1 - self.beta2**state["step_count"])

                    update = m_hat / (torch.sqrt(v_hat) + 1e-3)

                    param_vector.add_(precond_grad, alpha=-group["lr"])
                    # param_vector.add_(update, alpha=-group["lr"])
                    # Update gradient for analysis (optional)
                    p.grad.data = precond_grad.reshape(original_shape)
                    p.data = param_vector.reshape(original_shape)

        return loss