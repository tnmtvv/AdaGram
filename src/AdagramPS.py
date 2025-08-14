import torch

from typing import Optional, Dict, Any
from src.AdagramBase import AdaGram, AdaGramLogger

from line_profiler import profile


class AdaGramPS(AdaGram):
    """Adagram algorithm with projector splitting strategy"""
    def __init__(
        self,
        params,
        lr: float = 1.0,
        eps: float = 1e-10,
        weight_decay: float = 0,
        max_rank: Optional[int] = None,
        task: str = "LinReg",
        save_dir: str = "matrix_G",
        logger: Optional["AdaGramLogger"] = False,
        enable_logging: bool = False,
        save_matrix: bool = False,
    ):

        super().__init__(
            params=params,
            lr=lr,
            eps=eps,
            weight_decay=weight_decay,
            max_rank=max_rank,
            task=task,
            save_dir=save_dir,
            logger=logger,
            enable_logging=enable_logging,
            save_matrix=save_matrix,
        )
        self.save_dir = save_dir
        self.task_name = task

    @profile
    def reduce_rank_psi(self, delta_A, U_0, S_0, V_0):
        K_cur = U_0 @ S_0 + delta_A @ V_0
        U_cur, S_hat = torch.linalg.qr(K_cur)
        S_tild = S_hat - U_cur.T @ (delta_A @ V_0)
        L_cur = V_0 @ S_tild.T + delta_A.T @ U_cur
        V_cur, S_cur_T = torch.linalg.qr(L_cur)
        return U_cur, S_cur_T.T, V_cur
    

    @profile
    def one_rank_psi(self, b, g, u, s, v):
        # Ensure all inputs are 1D and on the same device/dtype
        g = g.flatten()
        u = u.flatten()
        s = s.flatten()
        v = v.flatten()

        # Precompute scalar products
        gu = torch.dot(g, s * u)       # P = u*s
        gv = torch.dot(g, v)           # g^T v
        vv = torch.dot(v, v)           # v^T v

        # Only one vector multiplication per use
        const = b * (gv - gu * vv)     # Scalar
        delta_av = const * g           # Vector

        # Compute K and norm efficiently
        K_cur = u * s + delta_av       # Vector
        K_norm = torch.sqrt(torch.dot(K_cur, K_cur))     # Scalar

        U_cur = K_cur / K_norm         # Unit vector

        # S_hat = K_norm, S_tild = S_hat - <U_cur, delta_av>
        S_tild = K_norm - torch.dot(U_cur, delta_av)  # Scalar

        gk = torch.dot(g, U_cur)                      # Scalar

        # delta_au is fully vectorized, reuses gu, gk
        delta_au = b * (g - v * gu) * gk              # Vector

        # L_cur etc
        L_cur = v * S_tild + delta_au                 # Vector
        L_norm = torch.sqrt(torch.dot(L_cur, L_cur))                    # Scalar
        V_cur = L_cur / L_norm                        # Unit vector

        # Return as column vectors if needed
        return U_cur.unsqueeze(1), L_norm, V_cur.unsqueeze(1)

    @profile
    def update_PQ(
        self,
        state: Dict[str, Any],
        beta: torch.Tensor,
        g_bar: torch.Tensor,
    ):

        beta_g = (beta * g_bar).reshape(-1, 1)
        g_bar_col = g_bar.reshape(-1, 1)

        reconstruct_error = torch.tensor(0, device=g_bar.device, dtype=g_bar.dtype)

        if "P" not in state:
            state["P"] = beta_g
            state["Q"] = g_bar_col


        elif not self.max_rank or (self.max_rank is not None and state["P"].shape[1] < self.max_rank):



            v_upd = g_bar_col - state["Q"] @ (state["P"].T @ g_bar_col) # update without matrices 

            state["P"] = torch.concat([state["P"], beta_g], dim=1)
            state["Q"] = torch.concat([state["Q"], v_upd], dim=1)
        
        elif self.max_rank is not None and state["P"].shape[1] >= self.max_rank:
            if "U" not in state:
                self._faster_svd(state)
                reconstruct_error = 0

            if self.max_rank == 1:

                if self.enable_logging:
                    prev_matrix = state["P"] @ state["Q"].T

                update = g_bar
                state["U"], state["S"], state["V"] = self.one_rank_psi(
                beta, update, state["U"], state["S"], state["V"]
            ) 
                state["P"] = state["U"] * state["S"]

            elif self.max_rank > 0:
                print("self.max_rank", self.max_rank)
                if self.enable_logging:
                    prev_matrix = state["P"] @ state["Q"].T

                g_bar_col = g_bar.reshape(-1, 1)
                g_p_proj = (g_bar @ state["P"]).reshape(-1) 

                update = (beta * g_bar_col) @ (g_bar - g_p_proj @ state["Q"].T).reshape(1, -1)


                state["U"], state["S"], state["V"] = self.reduce_rank_psi(
                    update, state["U"], state["S"], state["V"]
                )  # here all the matrices are not transposed
                state["P"] = state["U"] @ state["S"] # S matrix is not diagonal!

            
            state["Q"] = state["V"]

            if self.enable_logging:
                state["rec_target"] = prev_matrix + update

                if self.max_rank > 1:
                    reconstruct_error = torch.norm(
                        torch.abs(state["rec_target"] - state["U"] @ state["S"] @ state["V"].T)
                    ) / torch.norm(state["rec_target"])
                else:
                    reconstruct_error = torch.norm(
                        torch.abs(state["rec_target"] - state["S"] * state["U"] @ state["V"].T)
                    ) / torch.norm(state["rec_target"])



        return state["P"], state["Q"], reconstruct_error
