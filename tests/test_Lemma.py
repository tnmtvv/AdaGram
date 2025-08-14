import torch
import pytest
import libcontext

import numpy as np


def compute_alpha(
    g_bar_norm_sq: torch.Tensor, eps: float = 1e-10
) -> torch.Tensor:
    """Compute alpha_t that satisfies the equation (6) in the theorem."""
    return ((1 + g_bar_norm_sq).sqrt() - 1) / g_bar_norm_sq

def compute_beta(
    alpha: torch.Tensor, g_bar_norm_sq: torch.Tensor
) -> torch.Tensor:
    """Compute beta_t as defined in the theorem."""
    return alpha / (1 + alpha * g_bar_norm_sq)

def pq_Lt(g, P, Q, L_0_inv):
    if P is None:
        g_bar = L_0_inv * g
    else:
        g_bar = (L_0_inv * g) - P @ (Q.T @ (L_0_inv * g))
    g_bar_norm_sq = torch.dot(g_bar, g_bar)
    alpha = compute_alpha(g_bar_norm_sq, 1e-5)
    beta = compute_beta(alpha, g_bar_norm_sq)
    beta_g = (beta * g_bar).reshape(-1, 1)
    g_bar_col = g_bar.reshape(-1, 1)

    if P is not None:

        v_upd = (g_bar_col - Q @ (P.T @ g_bar_col))

        P = torch.cat([P, beta_g], dim=1)
        Q = torch.cat([Q, v_upd], dim=1)
    else:
        P = beta_g
        Q = g_bar_col

    return P, Q, g_bar


def gt_Lt(L_t, g):
    identity = torch.eye(g.shape[0], device=g.device, dtype=g.dtype)
    g_bar = torch.linalg.inv(L_t) @ g

    g_bar_norm_sq = torch.dot(g_bar, g_bar)
    alpha = compute_alpha(g_bar_norm_sq, 1e-5)
    L_t = L_t @ (identity + alpha * torch.ger(g_bar, g_bar))
    return L_t, g_bar


@pytest.mark.parametrize("dim, n_steps", [(4, 10), (6, 10)])
def test_pq_Lt_vs_gt_Lt(dim, n_steps):
    torch.manual_seed(42)
    eps = 1e-2
    print(eps)

    G = eps * torch.eye(dim)
    L_0 = (np.sqrt(eps))

    L_0_inv = 1 / L_0
    P = None
    Q = None
    # Generate a sequence of random g vectors
    gs = [torch.randn(dim) for _ in range(n_steps)]

    L_t = L_0 * torch.eye(dim)

    for i, g in enumerate(gs):
        # Update via pq_Lt
        P, Q, g_bar_pq = pq_Lt(g, P, Q, L_0_inv)
        # Update via gt_Lt
        L_t, g_bar_gt = gt_Lt(L_t, g)
        print("g_bar_gt", g_bar_gt)
        print("g_bar_pq", g_bar_pq)
        # Compare g_bar

        assert torch.allclose(
            g_bar_pq, g_bar_gt, atol=1e-4, rtol=1e-4
        ), f"g_bar mismatch at step {i}"
        print("Max abs diff:", torch.max(torch.abs(g_bar_pq - g_bar_gt)).item())

        # Print for debugging (optional)
        # print(f"Step {i}: g_bar_pq = {g_bar_pq}, g_bar_gt = {g_bar_gt}")
