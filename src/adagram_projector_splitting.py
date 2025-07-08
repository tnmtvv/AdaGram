import torch
from torch.optim import Optimizer
import math
import csv
import os
import traceback


class AdaGramPS(Optimizer):
    def __init__(
        self,
        params,
        lr=0.1,
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
        super(AdaGramPS, self).__init__(params, defaults)

        self.log_file = log_file
        self._initialize_csv()
        self.if_first = True
        self.max_rank = max_rank

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

    def _compute_alpha(self, g_bar_norm_sq, eps):
        """Compute alpha_t that satisfies the equation (6) in the theorem."""
        # 1 + alpha_t*||g_bar_t||^2 = (1 + ||g_bar_t||^2)^(1/2)
        # alpha_t:
        # alpha_t = ((1 + ||g_bar_t||^2)^(1/2) - 1) / ||g_bar_t||^2
        return ((1 + g_bar_norm_sq).sqrt() - 1) / (g_bar_norm_sq + eps)

    def _compute_beta(self, alpha, g_bar_norm_sq):
        """Compute beta_t as defined in the theorem."""
        return alpha / (1 + alpha * g_bar_norm_sq)

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

                identity = torch.eye(n, device=grad.device, dtype=grad.dtype)

                if len(state) == 0:
                    # state["Lt_inv"] = torch.eye(
                    #     n, device=grad.device, dtype=grad.dtype
                    # ) * math.sqrt(1 / eps)
                    state["step_count"] = 0  # Initialize step counter
                    # state["G_0"] = torch.eye(
                    #     n, device=grad.device, dtype=grad.dtype
                    # ) * math.sqrt(eps)
                    state["G"] = torch.eye(n, device=grad.device, dtype=grad.dtype)

                    g_bar = (
                        torch.eye(n, device=grad.device, dtype=grad.dtype)
                        # * math.sqrt(1 / eps)
                        @ grad_vector
                    )

                else:
                    g_bar = (
                        (identity - state["P"] @ state["Q"].t())
                        @ grad_vector
                        # * math.sqrt(1 / eps)
                    )

                g_bar_norm_sq = torch.dot(g_bar, g_bar)

                # equation (6)
                alpha = self._compute_alpha(g_bar_norm_sq, eps)

                # equation (7)
                beta = self._compute_beta(alpha, g_bar_norm_sq)

                beta_g = (beta * g_bar).reshape(-1, 1)
                g_bar_col = g_bar.reshape(-1, 1)

                state["G"] += torch.ger(grad_vector, grad_vector)

                if "P" in state:
                    update = (
                        beta
                        * torch.ger(g_bar, g_bar)
                        @ (identity - state["P"] @ state["Q"].T)
                    )
                else:
                    update = (
                        beta
                        * torch.ger(g_bar, g_bar)
                        # @ (identity - state["P"] @ state["Q"].T)
                    )

                if "P" not in state:
                    # print("here")
                    # print("'U' not in state")
                    state["P"] = beta_g
                    state["Q"] = g_bar_col

                    state["L_t"] = identity + alpha * torch.ger(g_bar, g_bar)
                    result = state["L_t"] @ state["L_t"].T
                    target = state["G"]
                    if not torch.allclose(result, target, atol=1e-3):
                        print("the first one")
                        # print("False")
                        # print("L_t @ L_t.T:\n", result)
                        # print("state['G']:\n", target)
                    else:
                        print("the first one")
                        # print("TRUE!!!")
                        # print("state['G']:\n", target)
                    error_norm = torch.norm(torch.abs(target - result)) / torch.norm(
                        target
                    )

                elif max_rank is not None and state["P"].shape[1] < max_rank:
                    identity = torch.eye(
                        state["Q"].shape[0], device=g_bar.device, dtype=g_bar.dtype
                    )
                    v_upd = ((identity - state["Q"] @ state["P"].T) @ g_bar).reshape(
                        -1, 1
                    )
                    state["P"] = torch.concat([state["P"], beta_g], dim=1)
                    state["Q"] = torch.concat([state["Q"], v_upd], dim=1)

                    identity = torch.eye(n, device=grad.device, dtype=grad.dtype)

                    state["L_t"] = state["L_t"] @ (
                        identity + alpha * torch.ger(g_bar, g_bar)
                    )
                    result = state["L_t"] @ state["L_t"].T
                    target = state["G"]
                    error_norm = torch.norm(torch.abs(target - result)) / torch.norm(
                        target
                    )
                    # print("norm target", torch.norm(target))

                elif max_rank is not None and state["P"].shape[1] >= max_rank:
                    if "U" not in state:
                        state["U"], state["S"], state["V"] = torch.linalg.svd(
                            state["P"] @ state["Q"].T
                        )
                        # state["S"] = torch.diag(state["S"][: self.max_rank])
                        # state["U"] = state["U"][:, : self.max_rank]
                        # state["V"] = state["V"][: self.max_rank, :].T
                        state["S"] = torch.diag(state["S"])
                        state["U"] = state["U"]
                        state["V"] = state["V"].T

                    else:
                        state["U"], state["S"], state["V"] = self.reduce_rank_psi(
                            update, state["U"], state["S"], state["V"]
                        )
                        # state["V"] = state["V"].T
                    # Q_1, S, Q_2 = torch.linalg.svd(state["S"])
                    # Uk = U[:, : self.max_rank]
                    # Sk = S
                    # Vk = V[: self.max_rank, :]
                    state["P"] = state["U"] @ state["S"]
                    state["Q"] = state["V"]

                    identity = torch.eye(n, device=grad.device, dtype=grad.dtype)
                    # L_t_inv = identity - state["P"] @ state["Q"].T
                    # L_t_p_inv = (identity - beta * torch.ger(g_bar, g_bar)) @ L_t_inv
                    # print("det", torch.linalg.det(L_t_p_inv))
                    state["L_t"] = state["L_t"] @ (
                        identity + alpha * torch.ger(g_bar, g_bar)
                    )
                    result = state["L_t"] @ state["L_t"].T
                    target = state["G"]
                    error_norm = torch.norm(torch.abs(target - result)) / torch.norm(
                        target
                    )

                    # print("norm target", torch.norm(target))

                rank_U = torch.linalg.matrix_rank(state["P"])
                rank_V = torch.linalg.matrix_rank(state["Q"])

                max_U = state["P"].max()
                max_V = state["Q"].max()

                min_U = state["P"].min()
                min_V = state["Q"].min()

                # Increment step counter
                state["step_count"] += 1

                self._log_to_csv(
                    state["step_count"],
                    param_idx,
                    torch.sqrt(torch.dot(grad_vector, grad_vector)),
                    torch.std(grad_vector),
                    beta,
                    group["lr"],
                    error_norm,
                    rank_U,
                    rank_V,
                    max_U,
                    min_U,
                    max_V,
                    min_V,
                    state["P"].shape,
                    state["Q"].shape,
                )

                precond_grad = g_bar / torch.sqrt(1 + g_bar_norm_sq)

                param_vector.add_(precond_grad, alpha=-group["lr"])
                p.data = param_vector.reshape(original_shape)

        return loss
