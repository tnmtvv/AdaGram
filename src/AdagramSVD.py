import torch

from typing import Optional, Dict, Any, Tuple

from src.AdagramBase import AdaGram, AdaGramLogger
from line_profiler import profile


class AdaGramFR(AdaGram):
    """AdaGramFR - AdaGram with Full Rank reduction using SVD"""

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
        save_matrix: bool = False,
    ):

        super().__init__(
            params=params,
            lr=lr,
            eps=eps,
            weight_decay=weight_decay,
            max_rank=max_rank,
            task=task,
            logger=logger,
            enable_logging=enable_logging,
            save_matrix=save_matrix,
        )

    @profile
    def reduce_rank_svd(
        self, M: torch.Tensor, max_rank=None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        has_nan = torch.isnan(M).any()
        has_inf = torch.isinf(M).any()

        if has_nan:
            print("Warning: Tensor contains NaN values")
        if has_inf:
            print("Warning: Tensor contains infinite values")

        U, S, Vh = torch.linalg.svd(M, full_matrices=False)

        if max_rank:
            U_k = U[:, :max_rank]
            S_k = S[:max_rank]
            V_k = Vh[:max_rank, :]

            return U_k, S_k, V_k
        else:
            return U, S, Vh
    @profile
    def update_PQ(
        self,
        state: Dict[str, Any],
        beta: torch.Tensor,
        g_bar: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        beta_g = (beta * g_bar).reshape(-1, 1)
        g_bar_col = g_bar.reshape(-1, 1)

        if "P" not in state:
            print("no P case")
            P = beta_g
            Q = g_bar_col
            reconstruct_error = torch.tensor(0.0)
        else:
            v_upd = (g_bar - state["Q"] @ (state["P"].T @ g_bar)).reshape(-1, 1)

            P = torch.cat([state["P"], beta_g], dim=1)
            Q = torch.cat([state["Q"], v_upd], dim=1)

            reconstruct_error = torch.tensor(0.0)

            if (
                self.max_rank is not None and P.shape[1] >= self.max_rank
            ) or not self.max_rank:
                state["rec_target"] = P @ Q.T

                state["U"], state["S"], state["V"] = self.reduce_rank_svd(
                    state["rec_target"], max_rank=self.max_rank
                )
                state["S"] = torch.diag(state["S"])

                P = state["U"] * state["S"]
                Q = state["V"].T

                if self.enable_logging:
                    reconstruct_error = torch.norm(
                        torch.abs(
                            state["rec_target"] - state["U"] @ state["S"] @ state["V"]
                        )
                    ) / torch.norm(state["rec_target"])

        return P, Q, reconstruct_error
