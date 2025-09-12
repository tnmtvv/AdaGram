import torch

from typing import Optional, Dict, Any, Tuple

from AdagramBase import AdaGram, AdaGramLogger


class AdaGramVanilla(AdaGram):
    def __init__(
        self,
        params,
        lr: float = 1.0,
        eps: float = 1e-10,
        weight_decay: float = 0,
        max_rank: Optional[int] = None,
        task: str = "LinReg",
        logger: Optional["AdaGramLogger"] = False,
        enable_logging: bool = True,
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
        )

    def update_PQ(
        self,
        state: Dict[str, Any],
        beta: torch.Tensor,
        g_bar: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        beta_g = (beta * g_bar).reshape(-1, 1)
        g_bar_col = g_bar.reshape(-1, 1)

        if "P" not in state:

            P = beta_g
            Q = g_bar_col
            reconstruct_error = torch.tensor(0.0)
        else:
            v_upd = (g_bar_col - state["Q"] @ (state["P"].T @ g_bar_col))

            P = torch.cat([state["P"], beta_g], dim=1)
            Q = torch.cat([state["Q"], v_upd], dim=1)
            reconstruct_error = torch.tensor(0.0)

        return P, Q, reconstruct_error
