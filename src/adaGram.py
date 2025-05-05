import torch
from torch.optim import Optimizer
import math


class AdaGram(Optimizer):
    """Implements the full-matrix version of AdaGrad algorithm using recursive factorization.

    This optimizer adapts the learning rate using the full matrix of outer
    products of gradients, capturing correlations between parameters.

    The implementation uses the recursive formula G_t = L_t L_t^T where L_t
    is updated efficiently at each step without computing the full matrix inverse.

    Args:
        params (iterable): iterable of parameters to optimize
        lr (float, optional): learning rate (default: 1.0)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-10)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        max_rank (int, optional): maximum rank to maintain for U and V matrices (default: None)
    """

    def __init__(self, params, lr=1.0, eps=1e-10, weight_decay=0, max_rank=None):
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps <= 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, eps=eps, weight_decay=weight_decay, max_rank=max_rank)
        super(AdaGram, self).__init__(params, defaults)

    def _compute_alpha(self, g_bar_norm_sq, eps):
        """Compute alpha_t that satisfies the equation (6) in the theorem."""
        # 1 + alpha_t*||g_bar_t||^2 = (1 + ||g_bar_t||^2)^(1/2)
        # Solving for alpha_t:
        # alpha_t = ((1 + ||g_bar_t||^2)^(1/2) - 1) / ||g_bar_t||^2
        return ((1 + g_bar_norm_sq).sqrt() - 1) / (g_bar_norm_sq + eps)

    def _compute_beta(self, alpha, g_bar_norm_sq):
        """Compute beta_t as defined in the theorem."""
        return alpha / (1 + alpha * g_bar_norm_sq)

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
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                original_shape = p.data.shape
                grad_vector = grad.reshape(-1)
                param_vector = p.data.reshape(-1)
                n = len(grad_vector)

                if len(state) == 0:
                    state["Lt"] = torch.eye(n, device=grad.device, dtype=grad.dtype)
                    state["Lt_inv"] = torch.eye(n, device=grad.device, dtype=grad.dtype)

                if group["weight_decay"] != 0:
                    grad_vector = grad_vector.add(
                        param_vector, alpha=group["weight_decay"]
                    )

                # Compute g_bar = Lt^-1 * g (equation from theorem)
                g_bar = state["Lt_inv"] @ grad_vector
                g_bar_norm_sq = torch.dot(g_bar, g_bar)

                # Compute alpha according to equation (6)
                alpha = self._compute_alpha(g_bar_norm_sq, eps)

                # Compute beta according to the formula below equation (7)
                beta = self._compute_beta(alpha, g_bar_norm_sq)

                # Reshape g_bar for outer product calculation
                g_bar_col = g_bar.reshape(-1, 1)

                # Update Lt according to equation (5): Lt+1 = Lt(I + alpha * g_bar * g_bar^T)
                outer_product = g_bar_col @ g_bar_col.T
                identity = torch.eye(n, device=grad.device, dtype=grad.dtype)
                state["Lt"] = state["Lt"] @ (identity + alpha * outer_product)

                # Update Lt_inv according to equation (7): Lt+1^-1 = (I - beta * g_bar * g_bar^T) * Lt^-1
                state["Lt_inv"] = (identity - beta * outer_product) @ state["Lt_inv"]

                # Compute preconditioned gradient according to equation (4)
                precond_grad = g_bar / torch.sqrt(1 + g_bar_norm_sq)

                # Update parameters
                param_vector.add_(precond_grad, alpha=-group["lr"])
                p.data = param_vector.reshape(original_shape)

        return loss
