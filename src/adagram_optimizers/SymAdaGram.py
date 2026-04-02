from mmap import MAP_PRIVATE
import torch
from torch.optim import Optimizer
import numpy as np
import os

from typing import Optional, Dict, Any, Tuple
from abc import ABC

from utils.Logger import AdaGramLogger
from line_profiler import profile

from AdagramBase import AdaGram
from AdagramPS import AdaGramPS

class SymAdaGram(AdaGramPS, AdaGram):
    """
    AdaGram optimizer with Adam momentum integration.
    
    Implements logic of https://arxiv.org/pdf/2010.02022.
    """

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
    
    # can I keep it like that 

    @profile
    def reduce_rank_bug(self, b, g, U_0, S_0, V_0):
        # --- Shared precomputations ---
        P = U_0 @ S_0
        vv = V_0.T @ V_0                           # (r, r)
        P_vv = P @ vv

        # --- Delta_A @ V_0  (K-step increment, same as before) ---


        brackets_1 = g.T @ V_0 - g.T @ P_vv
        delta_av = b * g[:, None] @ brackets_1[None, :]


        # --- Delta_A^T @ U_0  (L-step increment; BUG uses U_0, not U_1) ---
        brackets_2 = g.T - (g.T @ P) @ V_0.T
        gu = g.T @ U_0
        delta_aTu = b * brackets_2[:, None] @ gu[None, :]


        # --- K-step and L-step are INDEPENDENT (both use only U_0, V_0) ---
        K = U_0 @ S_0 + delta_av                 # (n, r)
        U_cur, _ = torch.linalg.qr(K)               # (n, r)


        L = V_0 @ S_0.T + delta_aTu              # (n, r)
        V_cur, _ = torch.linalg.qr(L)               # (n, r)


        # --- Overlap matrices (basis change from old to new subspace) ---
        M = U_cur.T @ U_0                            # (r, r)  U_1^T U_0
        N = V_cur.T @ V_0                            # (r, r)  V_1^T V_0


        P_qtv = P @ (V_0.T @ V_cur)


        # --- U_1^T Delta_A V_1 ---
        u1_T     = U_cur.T @ g                         # (r,)
        brackets_3     = g.T @ V_cur - g.T @ P_qtv                        # (r,)


        u1_da_v1 = b * u1_T[:, None] @ brackets_3[None, :]  # (r, r)


        S_cur = M @ S_0 @ N.T + u1_da_v1            # (r, r)


        return U_cur, S_cur, V_cur


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

            if self.enable_logging:
                prev_matrix = state["P"] @ state["Q"].T

            update = g_bar
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
