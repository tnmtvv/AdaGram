import torch
from torch.optim import Optimizer
import math


class AdaGramPS(Optimizer):
    def __init__(self, params, lr=1.0, eps=1e-10, weight_decay=0, max_rank=None):
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps <= 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, eps=eps, weight_decay=weight_decay, max_rank=max_rank)
        super(AdaGramPS, self).__init__(params, defaults)

    def _compute_alpha(self, g_bar_norm_sq, eps):
        """Compute alpha_t that satisfies the equation (6) in the theorem."""
        # 1 + alpha_t*||g_bar_t||^2 = (1 + ||g_bar_t||^2)^(1/2)
        # Solving for alpha_t:
        # alpha_t = ((1 + ||g_bar_t||^2)^(1/2) - 1) / ||g_bar_t||^2
        return ((1 + g_bar_norm_sq).sqrt() - 1) / (g_bar_norm_sq + eps)

    def _compute_beta(self, alpha, g_bar_norm_sq):
        """Compute beta_t as defined in the theorem."""
        return alpha / (1 + alpha * g_bar_norm_sq)

    def _reduce_rank(self, M, max_rank):
        U, S, V = torch.linalg.svd(M, full_matrices=False)
        U = U[:, :max_rank]
        S = S[:max_rank]
        V = V[:max_rank, :]
        return U @ torch.diag(S) @ V

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

                if len(state) == 0:
                    state["U"] = torch.zeros(n, 1, device=grad.device, dtype=grad.dtype)
                    state["V"] = torch.zeros(n, 1, device=grad.device, dtype=grad.dtype)
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

                if state["U"].shape[1] < max_rank:
                    state["U"] = torch.cat([state["U"], beta_g], dim=1)
                    state["V"] = torch.cat([state["V"], g_bar_col], dim=1)

                else:
                    state["U"] = self._reduce_rank(state["U"], max_rank)
                    state["V"] = self._reduce_rank(state["V"], max_rank)

                # Recompute Lt_inv with updated U and V
                identity = torch.eye(n, device=grad.device, dtype=grad.dtype)
                L0_inv = identity * math.sqrt(1.0 / eps)

                # Only compute UV^T if we have factors
                if state["U"].shape[1] > 0:
                    # For numerical stability, use a more stable approach
                    # Compute UV^T explicitly only if dimensions are reasonable
                    if n <= 10000 or state["U"].shape[1] <= 100:
                        UV_t = state["U"] @ state["V"].t()
                        state["Lt_inv"] = (identity - UV_t) @ L0_inv
                    else:
                        # For large dimensions, use a more efficient approach
                        # that avoids forming the full UV^T matrix
                        def lt_inv_matvec(x):
                            return L0_inv @ x - L0_inv @ (
                                state["U"] @ (state["V"].t() @ (L0_inv @ x))
                            )

                        # Apply the operator to standard basis vectors to form Lt_inv
                        # This is more efficient for large dimensions
                        state["Lt_inv"] = torch.stack(
                            [lt_inv_matvec(identity[:, i]) for i in range(n)], dim=1
                        )
                else:
                    state["Lt_inv"] = L0_inv

                # Update parameters
                precond_grad = g_bar / torch.sqrt(1 + g_bar_norm_sq)
                param_vector.add_(precond_grad, alpha=-group["lr"])
                p.data = param_vector.reshape(original_shape)

        return loss
