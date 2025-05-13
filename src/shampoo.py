import torch
from torch.optim import Optimizer
import math


class Shampoo(Optimizer):
    """Implements Shampoo optimizer for 2D parameters.

    Shampoo uses matrix-valued preconditioning to capture parameter correlations
    while maintaining computational efficiency compared to full-matrix methods.

    Args:
        params (iterable): iterable of parameters to optimize
        lr (float, optional): learning rate (default: 1.0)
        momentum (float, optional): momentum factor (default: 0.0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        eps (float, optional): term added to preconditioners for numerical stability (default: 1e-10)
        update_freq (int, optional): frequency of preconditioner updates (default: 1)
    """

    def __init__(
        self, params, lr=1.0, momentum=0.0, weight_decay=0, eps=1e-10, update_freq=1
    ):
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if eps <= 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if update_freq <= 0:
            raise ValueError(f"Invalid update frequency: {update_freq}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            eps=eps,
            update_freq=update_freq,
        )
        super(Shampoo, self).__init__(params, defaults)

        # Initialize step counter
        self.steps = 0

    def compute_matrix_power(self, matrix, power=-0.25):
        """Compute matrix raised to a power using SVD."""
        # Add small diagonal for numerical stability
        matrix = (
            matrix
            + torch.eye(matrix.size(0), device=matrix.device, dtype=matrix.dtype) * 1e-9
        )

        try:
            # Perform SVD
            U, S, Vh = torch.linalg.svd(matrix, full_matrices=False)

            # Ensure eigenvalues are positive
            S = torch.clamp(S, min=1e-10)

            # Compute power of singular values
            S_power = torch.pow(S, power)

            # Reconstruct the matrix
            return U @ torch.diag(S_power) @ Vh
        except RuntimeError:
            # Fallback to diagonal approximation if SVD fails
            diag = torch.diag(matrix)
            diag_power = torch.pow(torch.clamp(diag, min=1e-10), power)
            return torch.diag(diag_power)

    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        # Increment step counter
        self.steps += 1

        for group in self.param_groups:
            eps = group["eps"]
            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]
            update_freq = group["update_freq"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data

                # Apply weight decay if specified
                if weight_decay != 0:
                    grad = grad.add(p.data, alpha=weight_decay)

                # Get or initialize state
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0

                    # Initialize momentum buffer if needed
                    if momentum > 0:
                        state["momentum_buffer"] = torch.zeros_like(grad)

                # Apply momentum if specified
                if momentum > 0:
                    state["momentum_buffer"].mul_(momentum).add_(grad)
                    grad = state["momentum_buffer"]

                # Handle 2D parameters

                if len(p.data.shape) == 2:
                    m, n = p.data.shape

                    # Initialize preconditioners if needed
                    if "Lt" not in state:
                        state["Lt"] = eps * torch.eye(
                            m, device=grad.device, dtype=grad.dtype
                        )
                        state["Rt"] = eps * torch.eye(
                            n, device=grad.device, dtype=grad.dtype
                        )

                    # Update preconditioners periodically
                    if self.steps % update_freq == 0:
                        # Update left preconditioner: Lt = Lt + G_t * G_t^T
                        state["Lt"].add_(grad @ grad.T)

                        # Update right preconditioner: Rt = Rt + G_t^T * G_t
                        state["Rt"].add_(grad.T @ grad)

                    # Compute preconditioned gradient
                    left_precond = self.compute_matrix_power(state["Lt"], power=-0.25)
                    right_precond = self.compute_matrix_power(state["Rt"], power=-0.25)

                    # Apply preconditioned update
                    precond_grad = left_precond @ grad @ right_precond
                    p.data.add_(precond_grad, alpha=-lr)

                elif len(p.data.shape) == 1:
                    # Initialize preconditioner if needed
                    if "diag" not in state:
                        state["diag"] = eps * torch.ones_like(p.data)

                    # Update preconditioner
                    if self.steps % update_freq == 0:
                        state["diag"].add_(grad * grad)

                    # Apply preconditioned update (equivalent to AdaGrad)
                    precond_grad = grad / torch.sqrt(state["diag"])
                    p.data.add_(precond_grad, alpha=-lr)

                # Handle non-2D parameters with diagonal approximation
                else:
                    raise NotImplementedError(
                        "Shampoo optimizer currently only supports 2D parameters"
                    )
        return loss
