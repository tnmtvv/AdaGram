import torch
from torch.optim import Optimizer


from typing import Optional, Dict, Any, Tuple
from src.adagram_base import AdaGram, AdaGramLogger


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
        """
        Initialize AdaGramVanilla optimizer

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

    def update_PQ(
        self,
        state: Dict[str, Any],
        beta: torch.Tensor,
        g_bar: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Update P and Q matrices with rank reduction using SVD

        Args:
            state: Optimizer state dictionary
            beta: Beta coefficient
            g_bar: Preconditioned gradient
            grad_vector: Original gradient vector (unused in this implementation)
            alpha: Alpha coefficient (unused in this implementation)

        Returns:
            Tuple of (P, Q, reconstruction_error)
        """
        beta_g = (beta * g_bar).reshape(-1, 1)
        g_bar_col = g_bar.reshape(-1, 1)

        if "P" not in state:
            # First update - initialize P and Q
            P = beta_g
            Q = g_bar_col
            reconstruct_error = torch.tensor(0.0)
        else:
            # Subsequent updates - extend P and Q
            identity = torch.eye(
                state["Q"].shape[0], device=g_bar.device, dtype=g_bar.dtype
            )
            v_upd = ((identity - state["Q"] @ state["P"].T) @ g_bar).reshape(-1, 1)

            # Extend matrices

            P = torch.cat([state["P"], beta_g], dim=1)
            Q = torch.cat([state["Q"], v_upd], dim=1)
            reconstruct_error = torch.tensor(0.0)

        return P, Q, reconstruct_error
