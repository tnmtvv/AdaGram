import torch
from torch.optim import Optimizer
import csv
import os


class FullAdaGrad(Optimizer):
    def __init__(
        self,
        params,
        lr=1.0,
        eps=1e-10,
        weight_decay=0,
        log_file=f"results/loggs/adagram_full_adagrad.csv",
    ):
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps <= 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, eps=eps, weight_decay=weight_decay)
        super(FullAdaGrad, self).__init__(params, defaults)

        self.log_file = log_file

        # self._initialize_csv()
        self.eps = eps

    def _initialize_csv(self):
        """Initialize CSV file with headers if it doesn't exist"""
        if not os.path.isfile(self.log_file):
            with open(self.log_file, "a", newline="") as f:
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

    def _log_to_csv(
        self,
        step_count,
        param_id,
        grad_norm,
        grad_std,
        beta,
        lr,
        error_norm,
        rank_U,
        rank_V,
        max_U,
        min_U,
        max_V,
        min_V,
        U_shape,
        V_shape,
    ):
        """Log optimizer statistics to CSV file"""
        with open(self.log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    step_count,
                    param_id,
                    grad_norm.item(),
                    grad_std.item(),
                    beta.item(),
                    lr,
                    error_norm.item(),
                    rank_U.item(),
                    rank_V.item(),
                    max_U.item(),
                    min_U.item(),
                    max_V.item(),
                    min_V.item(),
                    U_shape[0],
                    U_shape[1],
                    V_shape[0],
                    V_shape[1],
                ]
            )

    def _compute_alpha(self, g_bar_norm_sq, eps=1e-10):
        """Compute alpha_t that satisfies the equation (6) in the theorem."""
        return ((1 + g_bar_norm_sq).sqrt() - 1) / (g_bar_norm_sq)

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

        with torch.no_grad():
            for group in self.param_groups:

                for param_idx, p in enumerate(group["params"]):
                    if p.grad is None:
                        continue

                    # print("p.grad.data.shape", p.grad.data.shape)
                    grad = p.grad.data
                    state = self.state[p]

                    original_shape = p.data.shape

                    grad_vector = grad.reshape(-1)
                    param_vector = p.data.reshape(-1)
                    n = len(grad_vector)

                    if len(state) == 0:
                        print("self.eps", self.eps)
                        print("grad_vector", grad_vector)
                        state["G"] = self.eps * torch.eye(
                            n, device=grad.device, dtype=grad.dtype
                        )
                        state["step_count"] = 0  # Initialize step counter

                    state["G"] += torch.ger(grad_vector, grad_vector)

                    eigenvals, eigenvecs = torch.linalg.eigh(state["G"])
                    # clamped_eigenvals = torch.clamp(eigenvals, min=0.0)
                    # sqrt_eigenvals = torch.sqrt(clamped_eigenvals + 0.0001)
                    sqr_G = eigenvecs @ torch.diag(eigenvals) @ eigenvecs.T

                    precond_grad = torch.linalg.inv(sqr_G) @ grad_vector
                        
                    param_vector.add_(precond_grad, alpha=-group["lr"])
                    p.data = param_vector.reshape(original_shape)
                    p.grad.data = precond_grad # for analysis
                    state["step_count"] += 1

        return loss
