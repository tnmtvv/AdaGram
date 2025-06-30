import torch
from torch.optim import Optimizer
import math
import csv
import os
import traceback


class AdaGramFR(Optimizer):
    def __init__(
        self,
        if_svd,
        params,
        lr=1.0,
        eps=1e-10,
        weight_decay=0,
        max_rank=None,
        log_file=f"results/adagram_logs.csv",
    ):
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps <= 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, eps=eps, weight_decay=weight_decay, max_rank=max_rank)
        super(AdaGramFR, self).__init__(params, defaults)

        # CSV logging setup
        self.full_svd = if_svd
        if self.full_svd:
            self.log_file = f"results/loggs/svd_adagram_logs.csv"
        else:
            self.log_file = f"results/loggs/nosvd_adagram_logs.csv"

        self._initialize_csv()
        self.if_first = True

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

    def _compute_alpha(self, g_bar_norm_sq, eps):
        """Compute alpha_t that satisfies the equation (6) in the theorem."""
        # 1 + alpha_t*||g_bar_t||^2 = (1 + ||g_bar_t||^2)^(1/2)
        # alpha_t:
        # alpha_t = ((1 + ||g_bar_t||^2)^(1/2) - 1) / ||g_bar_t||^2
        return ((1 + g_bar_norm_sq).sqrt() - 1) / (g_bar_norm_sq + eps)

    def _compute_beta(self, alpha, g_bar_norm_sq):
        """Compute beta_t as defined in the theorem."""
        return alpha / (1 + alpha * g_bar_norm_sq)

    def is_orthogonal(self, U, tolerance=1e-6):
        """Check if matrix U is orthogonal"""
        # Compute U^T @ U
        product = U.T @ U

        # Create identity matrix of same size
        identity = torch.eye(U.shape[1], device=U.device, dtype=U.dtype)

        # Check if they're approximately equal
        return torch.allclose(product, identity, atol=tolerance)

    def reduce_rank_psi(self, delta_A, U_0, S_0, V_0):
        """Standard SVD rank reduction"""
        K_cur = U_0 @ S_0 + delta_A @ V_0
        U_cur, S_hat = torch.linalg.qr(K_cur)
        S_tild = S_hat - U_cur.T @ delta_A @ V_0
        L_cur = V_0 @ S_tild.T + delta_A.T @ U_cur
        V_cur, S_cur_T = torch.linalg.qr(L_cur)
        return U_cur, S_cur_T.T, V_cur

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

            for param_idx, p in enumerate(group["params"]):
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                original_shape = p.data.shape

                grad_vector = grad.reshape(-1)
                param_vector = p.data.reshape(-1)
                n = len(grad_vector)

                update = grad_vector @ grad_vector.T

                identity = torch.eye(n, device=grad.device, dtype=grad.dtype)

                if len(state) == 0:
                    state["Lt_inv"] = torch.eye(
                        n, device=grad.device, dtype=grad.dtype
                    ) * math.sqrt(1 / eps)
                    state["step_count"] = 0  # Initialize step counter
                    state["G_0"] = torch.eye(
                        n, device=grad.device, dtype=grad.dtype
                    ) * math.sqrt(eps)
                    state["U"], state["S"], state["V"] = torch.linalg.svd(
                        state["G_0"] + update
                    )
                    g_bar = (
                        torch.eye(n, device=grad.device, dtype=grad.dtype)
                        * math.sqrt(1 / eps)
                        @ grad_vector
                    )

                elif "U" in state:
                    U, S, V = self.reduce_rank_psi(
                        update, state["U"], state["S"], state["V"]
                    )
                    state["U"] = U[:, : self.max_rank]
                    state["S"] = S[: self.max_rank]
                    state["V"] = V[: self.max_rank, :]
                    g_bar = (
                        (identity - state["U"] @ state["V"].t())
                        @ grad_vector
                        * math.sqrt(1 / eps)
                    )

                if "U" not in state or "V" not in state:
                    g_bar = (
                        torch.eye(n, device=grad.device, dtype=grad.dtype)
                        * math.sqrt(1 / eps)
                        @ grad_vector
                    )
                elif "U" in state and "V" in state:

                    # Define dimensions properly
                    n = grad_vector.size(0)
                    identity = torch.eye(n, device=grad.device, dtype=grad.dtype)

                    # Correct matrix operations
                    g_bar = (
                        (identity - state["U"] @ state["V"].t())
                        @ grad_vector
                        * math.sqrt(1 / eps)
                    )

                g_bar_norm_sq = torch.dot(g_bar, g_bar)

                # equation (6)
                alpha = self._compute_alpha(g_bar_norm_sq, eps)

                # equation (7)
                beta = self._compute_beta(alpha, g_bar_norm_sq)

                # beta_g = (beta * g_bar).reshape(-1, 1)
                # g_bar_col = g_bar.reshape(-1, 1)

                # if "U" not in state:
                #     # print("'U' not in state")
                #     state["U"] = beta_g
                #     state["V"] = g_bar_col
                # else:
                #     if max_rank is not None and state["U"].shape[1] >= max_rank:
                #         if not self.full_svd:
                #             self.reduce_rank_brand("U", beta_g, max_rank, p)
                #             self.reduce_rank_brand("V", g_bar_col, max_rank, p)
                #         else:
                #             state["U"] = torch.cat([state["U"], beta_g], dim=1)
                #             state["V"] = torch.cat([state["V"], g_bar_col], dim=1)
                #             state["U"] = self._reduce_rank(
                #                 M=state["U"], max_rank=max_rank
                #             )
                #             state["V"] = self._reduce_rank(
                #                 M=state["V"], max_rank=max_rank
                #             )
                #     elif state["U"].shape[1] < max_rank:
                #         state["U"] = torch.concat([state["U"], beta_g], dim=1)
                #         state["V"] = torch.concat([state["V"], g_bar_col], dim=1)

                rank_U = torch.linalg.matrix_rank(state["U"])
                rank_V = torch.linalg.matrix_rank(state["V"])

                max_U = state["U"].max()
                max_V = state["V"].max()

                min_U = state["U"].min()
                min_V = state["V"].min()

                # Increment step counter
                state["step_count"] += 1

                self._log_to_csv(
                    state["step_count"],
                    param_idx,
                    torch.sqrt(torch.dot(grad_vector, grad_vector)),
                    torch.std(grad_vector),
                    beta,
                    group["lr"],
                    rank_U,
                    rank_V,
                    max_U,
                    min_U,
                    max_V,
                    min_V,
                    state["U"].shape,
                    state["V"].shape,
                )

                precond_grad = g_bar / torch.sqrt(1 + g_bar_norm_sq)

                param_vector.add_(precond_grad, alpha=-group["lr"])
                p.data = param_vector.reshape(original_shape)

        return loss
