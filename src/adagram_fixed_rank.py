import torch
from torch.optim import Optimizer
import math


class AdaGramFR(Optimizer):
    def __init__(self, params, lr=1.0, eps=1e-10, weight_decay=0, max_rank=None):
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps <= 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, eps=eps, weight_decay=weight_decay, max_rank=max_rank)
        super(AdaGramFR, self).__init__(params, defaults)

    def _compute_alpha(self, g_bar_norm_sq, eps):
        """Compute alpha_t that satisfies the equation (6) in the theorem."""
        # 1 + alpha_t*||g_bar_t||^2 = (1 + ||g_bar_t||^2)^(1/2)
        # alpha_t:
        # alpha_t = ((1 + ||g_bar_t||^2)^(1/2) - 1) / ||g_bar_t||^2
        return ((1 + g_bar_norm_sq).sqrt() - 1) / (g_bar_norm_sq + eps)

    def _compute_beta(self, alpha, g_bar_norm_sq):
        """Compute beta_t as defined in the theorem."""
        return alpha / (1 + alpha * g_bar_norm_sq)

    def _reduce_rank(self, M, max_rank):
        U, S, Vh = torch.linalg.svd(M, full_matrices=False)

        # Project original matrix onto top max_rank components
        U_k = U[:, :max_rank]
        S_k = S[:max_rank]
        return U_k * S_k.unsqueeze(0)  # Shape: (rows, max_rank)

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
            max_rank = group.get("max_rank")

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                original_shape = p.data.shape
                grad_vector = grad.reshape(-1)
                param_vector = p.data.reshape(-1)
                n = len(grad_vector)

                identity = torch.eye(n, device=grad.device, dtype=grad.dtype)
                # L0_inv = identity * math.sqrt(1.0 / eps)

                if len(state) == 0:
                    state["U"] = torch.empty(n, 0, device=grad.device, dtype=grad.dtype)
                    state["V"] = torch.empty(n, 0, device=grad.device, dtype=grad.dtype)
                    state["Lt_inv"] = torch.eye(
                        n, device=grad.device, dtype=grad.dtype
                    ) * math.sqrt(1.0 / eps)
                    state["Sigma"] = torch.ones(
                        state["U"].shape[1], device=state["U"].device
                    )

                if group["weight_decay"] != 0:
                    grad_vector = grad_vector.add(
                        param_vector, alpha=group["weight_decay"]
                    )

                # g_bar = Lt^-1 * g (equation from theorem)
                g_bar = state["Lt_inv"] @ grad_vector
                g_bar_norm_sq = torch.dot(g_bar, g_bar)

                # equation (6)
                alpha = self._compute_alpha(g_bar_norm_sq, eps)

                # equation (7)
                beta = self._compute_beta(alpha, g_bar_norm_sq)

                beta_g = (beta * grad_vector).reshape(-1, 1)
                g_bar_col = g_bar.reshape(-1, 1)

                state["U"] = torch.cat([state["U"], beta_g], dim=1)
                state["V"] = torch.cat([state["V"], g_bar_col], dim=1)

                if state["U"].shape[1] > max_rank:
                    state["U"] = self._reduce_rank(state["U"], max_rank)
                    state["V"] = self._reduce_rank(state["V"], max_rank)
                    rank_U = torch.linalg.matrix_rank(state["U"])
                    rank_V = torch.linalg.matrix_rank(state["V"])
                    print("rank_U: ", rank_U)
                    print("rank_V: ", rank_V)
                    print("state['V'].max: ", state["V"].max())
                    print("state['U'].max: ", state["U"].max())

                UV_t = state["U"] @ state["V"].t()
                state["Lt_inv"] = (identity - UV_t) @ state["Lt_inv"]

                precond_grad = g_bar / torch.sqrt(1 + g_bar_norm_sq)
                param_vector.add_(precond_grad, alpha=-group["lr"])
                p.data = param_vector.reshape(original_shape)

        return loss
