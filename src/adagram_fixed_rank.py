import torch
from torch.optim import Optimizer
import math
import csv
import os
import traceback
import numpy as np
from typing import Optional, Dict, Any, Tuple

from src.adagram_base import AdaGram, AdaGramLogger


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
    ):
        """
        Initialize AdaGramFR optimizer

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

    def reduce_rank_svd(
        self, M: torch.Tensor, max_rank=None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Reduce rank using SVD decomposition with numerical stability checks

        Args:
            M: Matrix to decompose
            max_rank: Maximum rank to keep

        Returns:
            Tuple of (U_k, S_k, V_k) - reduced rank decomposition
        """
        # Check for numerical issues
        has_nan = torch.isnan(M).any()
        has_inf = torch.isinf(M).any()

        if has_nan:
            print("Warning: Tensor contains NaN values")
        if has_inf:
            print("Warning: Tensor contains infinite values")

        # Perform SVD
        U, S, Vh = torch.linalg.svd(M, full_matrices=False)

        # Keep only top max_rank components
        if max_rank:
            U_k = U[:, :max_rank]
            S_k = S[:max_rank]
            V_k = Vh[:max_rank, :]

            return U_k, S_k, V_k
        else:
            return U, S, Vh

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
            print("no P case")
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

            # Apply rank reduction if necessary
            if (
                self.max_rank is not None and P.shape[1] >= self.max_rank
            ) or not self.max_rank:
                state["rec_target"] = P @ Q.T

                state["U"], state["S"], state["V"] = self.reduce_rank_svd(
                    state["rec_target"], max_rank=self.max_rank
                )
                state["S"] = torch.diag(state["S"])

                # Update P and Q with reduced rank approximation
                P = state["U"] @ state["S"]
                Q = state["V"].T

                # Calculate reconstruction error
                reconstruct_error = torch.norm(
                    torch.abs(
                        state["rec_target"] - state["U"] @ state["S"] @ state["V"]
                    )
                ) / torch.norm(state["rec_target"])
                print("reconstruct_error", reconstruct_error)

        return P, Q, reconstruct_error
