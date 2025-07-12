import torch
import pytest


def reduce_rank_psi(delta_A, U_0, S_0, V_0):

    K_cur = U_0 @ S_0 + delta_A @ V_0

    U_cur, S_hat = torch.linalg.qr(K_cur)

    S_tild = S_hat - U_cur.T @ delta_A @ V_0

    L_cur = V_0 @ S_tild.T + delta_A.T @ U_cur

    V_cur, S_cur_T = torch.linalg.qr(L_cur)

    return U_cur, S_cur_T.T, V_cur


def check_reduce_rank_equivalence(prev_matrix, rtol=1e-1, atol=1e-1):
    """Check equivalence between power series and SVD methods"""
    # Get SVD of previous matrix
    u, s, v = torch.linalg.svd(prev_matrix, full_matrices=False)

    # vec = torch.randn(prev_matrix.shape[0]).reshape(1, -1)
    # update = vec.T @ vec
    update = torch.randn_like(prev_matrix)

    cur_matrix = prev_matrix + update

    U_ps, S_ps, V_ps = reduce_rank_psi(update, u, torch.diag(s), v.T)
    reconst_ps = U_ps @ S_ps @ V_ps.T

    U_svd, S_svd, V_svd = torch.linalg.svd(cur_matrix, full_matrices=False)
    reconst_svd = U_svd @ torch.diag(S_svd) @ V_svd

    print("cur_matrix\n", cur_matrix)
    print("svd_matrix\n", reconst_svd)
    print("reconst_ps\n", reconst_ps)

    # Check equivalence
    assert torch.allclose(
        torch.abs(reconst_ps), torch.abs(reconst_svd), rtol=rtol, atol=atol
    ), f"Reconstruction mismatch: max diff = {torch.max(torch.abs(reconst_ps - cur_matrix))}"


def reduce_rank_svd(M, max_rank):
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)
    U_k = U[:, :max_rank]
    S_k = S[:max_rank]
    V_k = Vh[:max_rank, :]
    return U_k, S_k, V_k  # Shape: (rows, max_rank)


def check_svd(P, Q, rank):
    beta_g = torch.randn(P.shape[0], 1)
    g_bar_col = torch.randn(Q.shape[0], 1)
    P = torch.cat([P, beta_g], dim=1)
    Q = torch.cat([Q, g_bar_col], dim=1)

    U, S, V = reduce_rank_svd(P @ Q.T, max_rank=rank + 1)
    P_new = U @ torch.diag(S)
    Q_new = V.T

    print("P_new @ Q_new.T:\n", P_new @ Q_new.T)
    print("P @ Q.T:\n", P @ Q.T)

    assert torch.allclose(torch.abs(P_new @ Q_new.T), torch.abs(P @ Q.T), atol=1e-1)


def check_qr_way(P, Q, rank):
    beta_g = torch.randn(P.shape[0], 1)
    g_bar_col = torch.randn(Q.shape[0], 1)

    P = torch.cat([P, beta_g], dim=1)
    Q = torch.cat([Q, g_bar_col], dim=1)
    Q_p, R_p = torch.linalg.qr(P)
    Q_q, R_q = torch.linalg.qr(Q)
    matrix = R_p @ R_q.T
    u, sigm, v = torch.linalg.svd(matrix, full_matrices=False)
    uk = u[:, :rank]
    sigm = torch.diag(sigm[:rank])
    vk = v[:rank]
    P_new = Q_p @ uk @ sigm
    Q_new = Q_q @ vk.T

    print("P_new @ Q_new.T:\n", P_new @ Q_new.T)
    print("P @ Q.T:\n", P @ Q.T)

    assert torch.allclose(torch.abs(P_new @ Q_new.T), torch.abs(P @ Q.T), atol=1e-1)


class TestReduceRankPS:
    """Test class for power series rank reduction"""

    def setup_method(self):
        """Setup for each test method"""
        torch.manual_seed(42)

    def test_reduce_rank_equivalence_square(self):
        """Test equivalence for square matrices"""
        prev_matrix = torch.randn(5, 5)
        check_reduce_rank_equivalence(prev_matrix, rtol=1e-5, atol=1e-3)

    # def test_reduce_rank_equivalence_qr(self):
    #     """Test equivalence for matrices"""
    #     for rank in [4]:
    #         P = torch.randn(5, 5)
    #         Q = torch.randn(5, 5)
    #         check_qr_way(P, Q, rank)
    #         check_svd(P, Q, rank)


if __name__ == "__main__":
    # Run only this specific test
    pytest.main([__file__, "-v", "--tb=short"])
