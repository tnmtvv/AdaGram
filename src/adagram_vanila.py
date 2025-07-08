import torch
from torch.optim import Optimizer
import math
import csv
import os
import traceback


class AdaGram(Optimizer):
    def __init__(
        self,
        params,
        lr=1.0,
        eps=1e-10,
        weight_decay=0,
        log_file=f"results/adagram_logs.csv",
    ):
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps <= 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, eps=eps, weight_decay=weight_decay)
        super(AdaGram, self).__init__(params, defaults)

        # CSV logging setup
        # if self.full_svd:
        #     self.log_file = f"results/loggs/svd_adagram_logs.csv"
        # else:
        #     self.log_file = f"results/loggs/nosvd_adagram_logs.csv"

        self.log_file = f"results/loggs/adagram_vanila.csv"

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
                        "error_norm",
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
        # 1 + alpha_t*||g_bar_t||^2 = (1 + ||g_bar_t||^2)^(1/2)
        # alpha_t:
        # alpha_t = ((1 + ||g_bar_t||^2)^(1/2) - 1) / ||g_bar_t||^2
        return ((1 + g_bar_norm_sq).sqrt() - 1) / (g_bar_norm_sq)

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

    def reduce_rank_svd(self, M, max_rank):

        U, S, Vh = torch.linalg.svd(M, full_matrices=False)
        U_k = U[:, :max_rank]
        S_k = S[:max_rank]
        V_k = Vh[:max_rank, :]
        return U_k, S_k, V_k  # Shape: (rows, max_rank)

    def reduce_rank_brand_matrix(self, matrix, new_column, rank):
        """
        Update rank-r SVD when a new column is appended to the original matrix.

        Parameters:
        - matrix: Original matrix (m x n)
        - new_column: New column to be appended (m,)
        - rank: Desired rank of approximation

        Returns:
        - U: Updated U matrix (m x r)
        - S: Updated singular values matrix (r x r)
        - V: Updated V matrix (n+1 x r)
        """
        m, n = matrix.shape
        r = min(min(matrix.shape), rank)

        # Compute SVD of original matrix
        U, S_vals, Vh = torch.linalg.svd(matrix, full_matrices=False)
        S = torch.diag(S_vals)

        # Project new column onto U
        U_t_new_col = U.T @ new_column
        new_column_proj = new_column - U @ U_t_new_col
        p_norm = torch.norm(new_column_proj)

        # Construct K matrix
        zeros_row = torch.zeros(1, S.shape[0], device=matrix.device, dtype=matrix.dtype)
        K_top = torch.cat([S, U_t_new_col.reshape(-1, 1)], dim=1)
        K_bottom = torch.cat([zeros_row, p_norm.unsqueeze(0).unsqueeze(1)], dim=1)
        K = torch.cat([K_top, K_bottom], dim=0)
        print("K", K.shape)

        # SVD of K
        Uk, Sigma_k_vals, Vk_t = torch.linalg.svd(K, full_matrices=False)
        Sigma_k = torch.diag(Sigma_k_vals[:r])
        Uk_r = Uk[:, :r]
        Vk_r = Vk_t[:r, :].T

        # Normalize new column projection
        if p_norm > 1e-5:
            new_col_normalized = new_column_proj / p_norm
        else:
            new_col_normalized = torch.zeros(
                m, device=matrix.device, dtype=matrix.dtype
            )

        # Update U and V
        U_concat = torch.cat([U, new_col_normalized.reshape(-1, 1)], dim=1)
        U_new = U_concat @ Uk_r

        V_concat = torch.cat(
            [Vh.T, torch.zeros(n, 1, device=matrix.device, dtype=matrix.dtype)], dim=1
        )
        last_row = torch.zeros(1, n + 1, device=matrix.device, dtype=matrix.dtype)
        last_row[0, -1] = 1
        V_concat = torch.cat([V_concat, last_row], dim=0)
        V_new = V_concat @ Vk_r

        return U_new, Sigma_k, V_new

    def reduce_rank_brand_usv(self, U, S, V, new_column, rank):
        """
        Update rank-r SVD when a new column is appended to the original matrix.

        Parameters:
        - matrix: Original matrix (m x n)
        - new_column: New column to be appended (m,)
        - rank: Desired rank of approximation

        Returns:
        - U: Updated U matrix (m x r)
        - S: Updated singular values matrix (r x r)
        - V: Updated V matrix (n+1 x r)
        """
        m = U.shape[0]
        n = V.shape[1]

        print("m", m)
        print("n", n)

        U_t_new_col = U.T @ new_column
        new_column_proj = new_column - U @ U_t_new_col
        p_norm = torch.norm(new_column_proj)

        # Construct K matrix
        zeros_row = torch.zeros(1, S.shape[0], device=U.device, dtype=U.dtype)
        K_top = torch.cat([S, U_t_new_col.reshape(-1, 1)], dim=1)
        K_bottom = torch.cat([zeros_row, p_norm.unsqueeze(0).unsqueeze(1)], dim=1)
        K = torch.cat([K_top, K_bottom], dim=0)

        # SVD of K
        Uk, Sigma_k_vals, Vk_t = torch.linalg.svd(K, full_matrices=False)
        Sigma_k = torch.diag(Sigma_k_vals[:rank])
        Uk_r = Uk[:, :rank]
        Vk_r = Vk_t[:rank, :].T

        # Normalize new column projection
        if p_norm > 1e-5:
            new_col_normalized = new_column_proj / p_norm
        else:
            new_col_normalized = torch.zeros(m, device=U.device, dtype=U.dtype)

        # Update U and V
        U_concat = torch.cat([U, new_col_normalized.reshape(-1, 1)], dim=1)
        U_new = U_concat @ Uk_r

        V_concat = torch.cat(
            [V.T, torch.zeros(n, 1, device=V.device, dtype=U.dtype)], dim=1
        )
        last_row = torch.zeros(1, n + 1, device=U.device, dtype=U.dtype)
        last_row[0, -1] = 1
        V_concat = torch.cat([V_concat, last_row], dim=0)
        V_new = V_concat @ Vk_r

        return U_new, Sigma_k, V_new

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

                # print("p.grad.data.shape", p.grad.data.shape)
                grad = p.grad.data
                state = self.state[p]

                original_shape = p.data.shape

                # print("grad.shape!!!!!!!!!!!", grad.shape)
                grad_vector = grad.reshape(-1)
                param_vector = p.data.reshape(-1)
                n = len(grad_vector)
                # print(param_vector)

                identity = torch.eye(n, device=grad.device, dtype=grad.dtype)

                if len(state) == 0:
                    # state["S"] = torch.eye(
                    #     max_rank, device=grad.device, dtype=grad.dtype
                    # )
                    state["G"] = torch.eye(n, device=grad.device, dtype=grad.dtype)
                    state["step_count"] = 0  # Initialize step counter
                    L_0 = torch.linalg.cholesky(state["G"], upper=False)
                    state["L_0_inv"] = torch.linalg.inv(L_0)

                if "P" not in state or "Q" not in state:
                    g_bar = (
                        torch.eye(n, device=grad.device, dtype=grad.dtype)
                        @ state["L_0_inv"]
                        # * math.sqrt(1 / torch.sqrt(torch.tensor(eps)))
                        @ grad_vector
                    )
                elif "P" in state and "Q" in state:

                    # Define dimensions properlys
                    n = grad_vector.size(0)
                    identity = torch.eye(n, device=grad.device, dtype=grad.dtype)

                    # Correct matrix operations
                    g_bar = (
                        (identity - state["P"] @ state["Q"].T)
                        @ state["L_0_inv"]
                        # torch.linalg.inv(state["L_t"])
                        @ grad_vector
                        # * math.sqrt(1 / torch.sqrt(torch.tensor(eps)))
                    )

                g_bar_norm_sq = torch.dot(g_bar, g_bar)

                # equation (6)
                alpha = self._compute_alpha(g_bar_norm_sq, eps)

                # equation (7)
                beta = self._compute_beta(alpha, g_bar_norm_sq)

                beta_g = (beta * g_bar).reshape(-1, 1)
                g_bar_col = g_bar.reshape(-1, 1)

                state["G"] += torch.ger(grad_vector, grad_vector)

                if "P" not in state:
                    # print("'U' not in state")
                    state["P"] = beta_g
                    state["Q"] = g_bar_col

                    state["L_t"] = identity + alpha * torch.ger(g_bar, g_bar)
                    result = state["L_t"] @ state["L_t"].T
                    target = state["G"]
                    # if not torch.allclose(result, target, atol=1e-3):
                    #     print("the first one")
                    #     print("False")
                    #     # print("L_t @ L_t.T:\n", result)
                    #     # print("state['G']:\n", target)
                    # else:
                    #     print("the first one")
                    #     print("TRUE!!!")
                    #     # print("state['G']:\n", target)
                    error_norm = torch.norm(torch.abs(target - result)) / torch.norm(
                        target
                    )

                else:
                    identity = torch.eye(
                        state["Q"].shape[0], device=g_bar.device, dtype=g_bar.dtype
                    )
                    v_upd = ((identity - state["Q"] @ state["P"].T) @ g_bar).reshape(
                        -1, 1
                    )
                    state["P"] = torch.concat([state["P"], beta_g], dim=1)
                    state["Q"] = torch.concat([state["Q"], v_upd], dim=1)

                    state["L_t"] = state["L_t"] @ (
                        identity + alpha * torch.ger(g_bar, g_bar)
                    )
                    result = state["L_t"] @ state["L_t"].T
                    target = state["G"]
                    # if not torch.allclose(result, target, atol=1e-3):
                    #     print("the first one")
                    #     print("False")
                    #     # print("L_t @ L_t.T:\n", result)
                    #     # print("state['G']:\n", target)
                    # else:
                    #     # print("the first one")
                    #     # print("TRUE!!!")
                    #     # print("state['G']:\n", target)
                    error_norm = torch.norm(torch.abs(target - result)) / torch.norm(
                        target
                    )

                if "P" in state and "Q" in state:

                    rank_U = torch.linalg.matrix_rank(state["P"])
                    rank_V = torch.linalg.matrix_rank(state["Q"])

                    max_U = state["P"].max()
                    max_V = state["Q"].max()

                    min_U = state["P"].min()
                    min_V = state["Q"].min()

                else:
                    rank_U = rank_V = max_U = max_V = min_U = min_V = torch.tensor(0)

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

                # precond_grad = g_bar / torch.sqrt(1 + g_bar_norm_sq)
                # sqr_G = torch.linalg.cholesky(state["G"], upper=False)
                precond_grad = torch.linalg.inv(state["L_t"]) @ grad_vector

                param_vector.add_(precond_grad, alpha=-group["lr"])
                p.data = param_vector.reshape(original_shape)

        return loss
