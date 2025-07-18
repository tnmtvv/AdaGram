import torch
from torch.optim import Optimizer
import math
import csv
import os

from typing import Optional, Dict, Any, Tuple
from src.adagram_base import AdaGram, AdaGramLogger


class AdaGramPS(AdaGram):

    def __init__(
        self,
        params,
        lr: float = 1.0,
        eps: float = 1e-10,
        weight_decay: float = 0,
        max_rank: Optional[int] = None,
        task: str = "LinReg",
        logger: Optional["AdaGramLogger"] = None,
        enable_logging: bool = True,
    ):
        """
        Initialize AdaGramPS optimizer

        Args:
            params: Model parameters
            lr: Learning rate
            eps: Epsilon for numerical stability
            weight_decay: Weight decay factor
            max_rank: Maximum rank for approximation
            task: Task name
            logger: Custom logger instance
            enable_logging: Whether to enable logging
        """
        # Call parent constructor with all required parameters
        super().__init__(
            params=params,
            lr=lr,
            eps=eps,
            weight_decay=weight_decay,
            max_rank=max_rank,
            task=task,
            logger=logger,
            enable_logging=enable_logging,
        )

    def reduce_rank_psi(self, delta_A, U_0, S_0, V_0):

        K_cur = U_0 @ S_0 + delta_A @ V_0
        U_cur, S_hat = torch.linalg.qr(K_cur)
        S_tild = S_hat - U_cur.T @ delta_A @ V_0
        L_cur = V_0 @ S_tild.T + delta_A.T @ U_cur
        V_cur, S_cur_T = torch.linalg.qr(L_cur)
        return U_cur, S_cur_T.T, V_cur

    def update_PQ(
        self,
        state: Dict[str, Any],
        beta: torch.Tensor,
        g_bar: torch.Tensor,
    ):

        beta_g = (beta * g_bar).reshape(-1, 1)
        g_bar_col = g_bar.reshape(-1, 1)

        reconstruct_error = torch.tensor(0)

        if "P" not in state:
            # print("here")
            # print("'U' not in state")
            state["P"] = beta_g
            state["Q"] = g_bar_col

        elif self.max_rank is not None and state["P"].shape[1] < self.max_rank:
            identity = torch.eye(
                state["Q"].shape[0], device=g_bar.device, dtype=g_bar.dtype
            )
            v_upd = ((identity - state["Q"] @ state["P"].T) @ g_bar).reshape(-1, 1)
            state["P"] = torch.concat([state["P"], beta_g], dim=1)
            state["Q"] = torch.concat([state["Q"], v_upd], dim=1)

        elif not self.max_rank or (
            self.max_rank is not None and state["P"].shape[1] >= self.max_rank
        ):
            if "U" not in state:
                state["U"], state["S"], state["V"] = torch.linalg.svd(
                    state["P"] @ state["Q"].T
                )

                if self.max_rank:
                    state["U"] = state["U"][:, : self.max_rank]
                    state["S"] = torch.diag(state["S"][: self.max_rank])
                    state["V"] = state["V"][: self.max_rank, :].T
                else:
                    state["S"] = torch.diag(state["S"])
                    state["U"] = state["U"]
                    state["V"] = state["V"].T
                reconstruct_error = torch.norm(
                    torch.abs(
                        state["P"] @ state["Q"].T
                        - state["U"] @ state["S"] @ state["V"].T
                    )
                ) / torch.norm(state["P"] @ state["Q"].T)

                print("first_error", reconstruct_error)

            identity = torch.eye(
                state["P"].shape[0], device=g_bar.device, dtype=g_bar.dtype
            )
            update = (
                beta * torch.ger(g_bar, g_bar) @ (identity - state["P"] @ state["Q"].T)
            )
            prev_matrix = state["P"] @ state["Q"].T

            state["U"], state["S"], state["V"] = self.reduce_rank_psi(
                update, state["U"], state["S"], state["V"]
            )  # here all the matrices are not transposed

            state["rec_target"] = prev_matrix + update

            reconstruct_error = torch.norm(
                torch.abs(state["rec_target"] - state["U"] @ state["S"] @ state["V"].T)
            ) / torch.norm(state["rec_target"])

            state["P"] = state["U"] @ state["S"]
            state["Q"] = state["V"]
            print("reconstruct_error", reconstruct_error)
        return state["P"], state["Q"], reconstruct_error
