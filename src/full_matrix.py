import torch
from torch.optim import Optimizer
import math


class FullMatrixAdaGrad(Optimizer):
    """Implements the full-matrix version of AdaGrad algorithm.

    This optimizer adapts the learning rate using the full matrix of outer
    products of gradients, capturing correlations between parameters.

    Args:
        params (iterable): iterable of parameters to optimize
        lr (float, optional): learning rate (default: 1.0)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-10)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
    """

    def __init__(self, params, lr=1.0, eps=1e-10, weight_decay=0):
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps <= 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, eps=eps, weight_decay=weight_decay)
        super(FullMatrixAdaGrad, self).__init__(params, defaults)

    def compute_matrix_power(self, matrix, power=-0.25):
        """Compute matrix raised to a power using SVD."""

        matrix = (
            matrix
            + torch.eye(matrix.size(0), device=matrix.device, dtype=matrix.dtype)
            * 1e-9  # for numerical stability
        )

        try:
            U, S, Vh = torch.linalg.svd(matrix, full_matrices=False)
            S = torch.clamp(S, min=1e-10)
            S_power = torch.pow(S, power)
            return U @ torch.diag(S_power) @ Vh

        except RuntimeError as e:
            print(f"SVD failed: {e}")
            print(
                "Warning: Using diagonal approximation which is only correct for diagonal matrices"
            )
            diag = torch.diagonal(matrix)
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

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                original_shape = p.data.shape
                grad_vector = grad.reshape(-1)
                param_vector = p.data.reshape(-1)

                if len(state) == 0:
                    state["G"] = torch.zeros(
                        len(grad_vector),
                        len(grad_vector),
                        device=grad.device,
                        dtype=grad.dtype,
                    )

                if group["weight_decay"] != 0:
                    grad_vector = grad_vector.add(
                        param_vector, alpha=group["weight_decay"]
                    )

                outer_product = grad_vector.unsqueeze(1) @ grad_vector.unsqueeze(0)
                state["G"].add_(outer_product)

                # G^(1/2) - the square root of G
                G_inv_sqrt = self.compute_matrix_power(state["G"], power=-0.5)

                # update: -lr * G^(-1/2) * gradient
                update = -group["lr"] * (G_inv_sqrt @ grad_vector)

                param_vector.add_(update)

                p.data = param_vector.reshape(original_shape)

        return loss
