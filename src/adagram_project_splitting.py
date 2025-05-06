import torch
from torch.optim import Optimizer
import math


class AdaGramPS(Optimizer):
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
                    state["U"] = torch.zeros(n, 0, device=grad.device, dtype=grad.dtype)
                    state["V"] = torch.zeros(n, 0, device=grad.device, dtype=grad.dtype)
                    state["Lt"] = torch.eye(
                        n, device=grad.device, dtype=grad.dtype
                    ) / math.sqrt(eps)

                if group["weight_decay"] != 0:
                    grad_vector = grad_vector.add(
                        param_vector, alpha=group["weight_decay"]
                    )

                # Compute g_bar = L_t^(-1) * g_t+1
                # Using the recursive formula from equation (8) in Lemma 1:
                # L_t^(-1) = (I - U_t * V_t^T) * L_0^(-1)

                # First apply L_0^(-1) = (1/sqrt(eps)) * I
                g_bar = state["Lt"] @ grad_vector

                # # Then apply (I - U_t * V_t^T)
                # if state['U'].shape[1] > 0:  # Check if U_t is not empty
                #     g_bar = g_bar - state['U'] @ (state['V'].t() @ g_bar)

                # Compute ||g_bar||^2
                g_bar_norm_sq = torch.dot(g_bar, g_bar)

                # Compute alpha_t according to equation (6)
                alpha = self._compute_alpha(g_bar_norm_sq, eps)

                # Compute beta_t
                beta = self._compute_beta(alpha, g_bar_norm_sq)

                # Update U_{t+1} and V_{t+1} according to equation (9)
                # U_{t+1} = [U_t  beta_{t+1}*g_{t+1}]
                # V_{t+1} = [V_t  g_bar_{t+1}]

                # Scale beta_g by L_0^(-1) since it's applied to g_{t+1}
                beta_g = beta * g_bar

                # Reshape to column vectors for concatenation
                beta_g = beta_g.reshape(-1, 1)
                g_bar_col = g_bar.reshape(-1, 1)

                second_part = torch.eye() - state["V"] @ state["U"].T

                # Update U and V
                state["U"] = torch.cat([state["U"], beta_g], dim=1)
                state["V"] = torch.cat([state["V"], second_part @ g_bar_col], dim=1)

                # Limit the rank if max_rank is specified
                max_rank = group.get("max_rank")
                if max_rank is not None and state["U"].shape[1] > max_rank:
                    state["U"] = state["U"][:, -max_rank:]
                    state["V"] = state["V"][:, -max_rank:]

                state["Lt"] = (torch.eye() - state["V"] @ state["U"].T) @ state["Lt"]

                # Compute the preconditioned gradient using equation (4):
                # L_{t+1}^(-1) * g_{t+1} = (1/sqrt(1 + ||L_t^(-1)*g_{t+1}||^2)) * L_t^(-1)*g_{t+1}
                precond_grad = g_bar / torch.sqrt(1 + g_bar_norm_sq)
                param_vector.add_(-precond_grad, alpha=-group["lr"])

                p.data = param_vector.reshape(original_shape)

        return loss
