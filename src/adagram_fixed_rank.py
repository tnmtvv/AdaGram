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
        log_file="adagram_logs.csv",
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
        self.log_file = log_file
        self._initialize_csv()

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
                        "Lt_norm",
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

    def _reduce_rank(self, M, max_rank):
        """Standard SVD rank reduction"""
        try:
            U, S, Vh = torch.linalg.svd(M, full_matrices=False)
            U_k = U[:, :max_rank]
            S_k = S[:max_rank]
            V_k = Vh[:max_rank, :]
            return (U_k @ torch.diag(S_k) @ Vh[:max_rank, :])[
                :, :max_rank
            ]  # Shape: (rows, max_rank)
        except Exception as e:
            print("reduce rank error: ", e)
            print(M)
            torch.save(M, "error_matrix.pt")
            return None

    def reduce_rank(self, name, new_column, r, p):
        """
        Update rank-r SVD when a new column is appended to the original matrix using PyTorch.

        Parameters:
        - name: Name of the matrix in state
        - new_column: New column to be appended (m,)
        - r: Desired rank of approximation
        - p: Parameter index

        Returns:
        - Updated U matrix stored in state[name]
        """
        state = self.state[p]
        min_shape = min(state[name].shape)
        r = min(min_shape, r)

        # Initialize sigma if not exists
        if "sigma_" + name not in state:
            print("here")
            _, state["sigma_" + name], _ = torch.linalg.svd(
                state[name], full_matrices=False
            )
            state["sigma_" + name] = state["sigma_" + name][:r]

        m = state[name].shape[0]

        # Step 1: Project new column onto orthogonal complement of current U basis
        new_column_proj = new_column - state[name] @ (state[name].T @ new_column)
        print("new_column_proj", new_column_proj.shape)

        # Step 2: Compute the norm of the residual (orthogonal component)
        p_norm = torch.norm(new_column_proj)

        # Step 3: Form the intermediate matrix K using torch operations
        diag_sigma = torch.diag(state["sigma_" + name])
        U_t_new_col = state[name].T @ new_column

        # Create zeros row with correct dimensions
        zeros_row = torch.zeros(
            1,
            len(state["sigma_" + name]),
            device=state[name].device,
            dtype=state[name].dtype,
        )

        print("U_t_new_col.unsqueeze(1).shape", U_t_new_col.reshape(-1, 1).shape)
        print("diag_sigma.shape", diag_sigma.shape)
        print(f"state[{name}]", state[name].shape)

        K_top = torch.cat([diag_sigma, U_t_new_col.reshape(-1, 1)], dim=1)
        K_bottom = torch.cat([zeros_row, p_norm.unsqueeze(0).unsqueeze(1)], dim=1)
        K = torch.cat([K_top, K_bottom], dim=0)

        # Step 4: Compute SVD of K
        Uk, Sigma_k, _ = torch.linalg.svd(K, full_matrices=False)

        # Step 5: Truncate to rank r
        Uk_r = Uk[:, :r]
        state["sigma_" + name] = Sigma_k[:r]

        # Step 6: Update U matrix
        if p_norm > 1e-10:
            new_col_normalized = new_column_proj / p_norm
        else:
            new_col_normalized = torch.zeros(
                m, device=state[name].device, dtype=state[name].dtype
            )

        # Concatenate and update
        U_concat = torch.cat([state[name], new_col_normalized.reshape(-1, 1)], dim=1)
        state[name] = U_concat @ Uk_r

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
                    state["Lt_inv"] = torch.eye(
                        n, device=grad.device, dtype=grad.dtype
                    ) * math.sqrt(1 / eps)
                    state["step_count"] = 0  # Initialize step counter

                # if group["weight_decay"] != 0:
                #     grad_vector = grad_vector.add(
                #         param_vector, alpha=group["weight_decay"]
                #     )

                # g_bar = Lt^-1 * g (equation from theorem)
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

                beta_g = (beta * g_bar).reshape(-1, 1)
                g_bar_col = g_bar.reshape(-1, 1)

                if "U" not in state:
                    state["U"] = beta_g
                    state["V"] = g_bar_col
                else:
                    state["U"] = torch.cat([state["U"], beta_g], dim=1)
                    state["V"] = torch.cat([state["V"], g_bar_col], dim=1)

                if max_rank is not None and state["U"].shape[1] == max_rank:
                    if not self.full_svd:
                        "not full svd"
                        if state["U"].shape[0] >= max_rank:
                            self.reduce_rank("U", beta_g, max_rank, p)
                            self.reduce_rank("V", g_bar_col, max_rank, p)
                        else:
                            self._reduce_rank(state["U"], state["U"].shape[0])
                            self._reduce_rank(state["V"], state["U"].shape[0])
                    else:
                        state["U"] = self._reduce_rank(M=state["U"], max_rank=max_rank)
                        state["V"] = self._reduce_rank(M=state["V"], max_rank=max_rank)

                # metrics for logging
                try:
                    rank_U = torch.linalg.matrix_rank(state["U"])
                except Exception as e:
                    # Save problematic matrix and return error
                    print(f"state u, param_idx {param_idx}")
                    print("g_bar_norm_sq", g_bar_norm_sq)
                    print("eps", eps)
                    print("alpha", alpha)
                    print("beta", beta)
                    print("grad", grad_vector)
                    print("beta_g: (beta * grad_vector)", beta_g)
                    print("g_bar", g_bar)
                    print("state['Lt_inv']", state["Lt_inv"])
                    print("error", e)
                    print("state u: ", state["U"])
                    print("data", p.data)

                    # return

                try:
                    rank_V = torch.linalg.matrix_rank(state["V"])
                except Exception as e:
                    # Save problematic matrix and return error
                    print(f"state v, param_idx {param_idx}")
                    print("g_bar_col", g_bar_col)
                    print("beta", beta)
                    print("error", e)
                    print("state v: ", state["V"])

                    print("data", p.data)
                    print("grad", grad_vector)
                    # return

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

                # G^{-1/2}_t * g_(t + 1) ???
                precond_grad = g_bar / torch.sqrt(1 + g_bar_norm_sq)

                # print("update: ", -group["lr"] * precond_grad)
                # print("-----------------")
                param_vector.add_(precond_grad, alpha=-group["lr"])
                p.data = param_vector.reshape(original_shape)

        return loss
