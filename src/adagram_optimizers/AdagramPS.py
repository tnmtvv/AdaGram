import torch

# import libcontext
from typing import Optional, Dict, Any
from AdagramBase import AdaGram
from utils.Logger import AdaGramLogger


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
        alpha = None,
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
        self.alpha = alpha

    # @profile
    # def reduce_rank_psi(self, delta_A, U_0, S_0, V_0):
    #     if self.alpha:
    #         S_0 = self.alpha * S_0
    #         delta_A = (2 - self.alpha) * delta_A
            
    #         print("alpha", self.alpha)
    #     K_cur = U_0 @ S_0 + delta_A @ V_0
    #     U_cur, S_hat = torch.linalg.qr(K_cur)
    #     S_tild = S_hat - U_cur.T @ (delta_A @ V_0)
    #     L_cur = V_0 @ S_tild.T + delta_A.T @ U_cur
    #     V_cur, S_cur_T = torch.linalg.qr(L_cur)
    #     return U_cur, S_cur_T.T, V_cur


    # @profile
    # def reduce_rank_psi(self, b, g, U_0, S_0, V_0):
    #     vv = V_0.T @ V_0  # r x r
    #     g_us = g.T @ (U_0 @ S_0) # 1 x r
    #     gv = g.T @ V_0 # 1 x r

    #     l_r = b * (gv - g_us @ vv) # 1 x r
    #     delta_av = g @ l_r


    #     K_cur = U_0 @ S_0  + delta_av # n x r

    #     U_cur, S_hat = torch.linalg.qr(K_cur)
    #     S_tild = S_hat - U_cur.T @ (g @ l_r)

    #     delta_au = b * (g - V_0 @ S_0.T @ (U_0.T @ g)) @ (g.T @ U_cur)
    #     L_cur = V_0 @ S_tild.T + delta_au

    #     V_cur, S_cur_T = torch.linalg.qr(L_cur)
    #     return U_cur, S_cur_T.T, V_cur
    

    @profile
    def reduce_rank_psi(self, b, g, U_0, S_0, V_0):
        # Shapes assumed:
        # U_0: (n, r), S_0: (r, r), V_0: (n, r), g: (n,), b: scalar

        vv   = V_0.T @ V_0                 # (r, r)
        g_us = g @ (U_0 @ S_0)             # (r,)
        gv   = g @ V_0                     # (r,)

        l_r = b * (gv - g_us @ vv)         # (r,)
        delta_av = g[:, None] @ l_r[None, :]   # (n, r) outer product
        # K step
        K_cur = (U_0 @ S_0) + delta_av     # (n, r)
        U_cur, S_hat = torch.linalg.qr(K_cur)

        S_tild = S_hat - U_cur.T @ delta_av    # (r, r)

        t = g @ U_cur                     
        w = g - V_0 @ (S_0.T @ (U_0.T @ g))# (n,)
        delta_au = b * w[:, None] @ t[None, :] # (n, r) outer product

        # L step
        L_cur = (V_0 @ S_tild.T) + delta_au     # (n, r)
        V_cur, S_cur_T = torch.linalg.qr(L_cur)

        return U_cur, S_cur_T.T, V_cur


    @profile
    def one_rank_psi(self, b, g, u, s, v):
        # Ensure all inputs are 1D and on the same device/dtype
        g = g.flatten()
        u = u.flatten()
        s = s.flatten()
        v = v.flatten()

        if self.alpha:
            alpha = self.alpha 
        else: 
            alpha = 1 

        # Precompute scalar products
        g_us = torch.dot(g, s * u)       # P = u*s
        gv = torch.dot(g, v)           # g^T v
        vv = torch.dot(v, v)           # v^T v

        # Only one vector multiplication per use
        const = b * (gv - g_us * vv)     # Scalar
        delta_av = const * g           # Vector

        # Compute K and norm efficiently
        # s = alpha * s
        # delta_av = (2 - alpha) * delta_av

        K_cur = u * s + delta_av       # Vector
        K_norm = torch.sqrt(torch.dot(K_cur, K_cur))     # Scalar

        U_cur = K_cur / K_norm         # Unit vector

        # S_hat = K_norm, S_tild = S_hat - <U_cur, delta_av>
        S_tild = K_norm - torch.dot(U_cur, delta_av)  # Scalar

        gk = torch.dot(g, U_cur)                      # Scalar

        # delta_au is fully vectorized, reuses gu, gk
        delta_au = b * (g - v * g_us) * gk              # Vector

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
                print("reducing rank with old method")
                state["U"], state["S"], state["V"] = self.one_rank_psi(
                beta, update, state["U"], state["S"], state["V"]
            ) 
                state["P"] = state["U"] * state["S"]

            elif self.max_rank > 0:
                if self.enable_logging:
                    prev_matrix = state["P"] @ state["Q"].T

                # g_bar_col = g_bar.reshape(-1, 1)
                # g_p_proj = (g_bar @ state["P"]).reshape(-1) 

                # update = (beta * g_bar_col) @ (g_bar - g_p_proj @ state["Q"].T).reshape(1, -1)

                print("reducing rank with new method")
                state["U"], state["S"], state["V"] = self.reduce_rank_psi(
                    beta, g_bar, state["U"], state["S"], state["V"]
                )  # here all the matrices are not transposed
                state["P"] = state["U"] @ state["S"] # S matrix is not diagonal!

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
            
            state["Q"] = state["V"]

        return state["P"], state["Q"], reconstruct_error
