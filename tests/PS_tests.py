import torch
import pytest


def reduce_rank_ps(A_cur, A_prev, U_0, S_0, V_0):
    # Check dimensions
    m, n = A_cur.shape
    r = U_0.shape[1]
    assert U_0.shape == (m, r)
    assert S_0.shape == (r, r)
    assert V_0.shape == (n, r)

    delta_A = A_cur - A_prev

    # Step 1: Update U
    K_cur = U_0 @ S_0 + delta_A @ V_0
    U_cur, S_hat = torch.linalg.qr(K_cur)

    # Step 2: Mid-step for S
    S_tild = S_hat - U_cur.T @ delta_A @ V_0

    # Step 3: Update V
    L_cur = V_0 @ S_tild.T + delta_A.T @ U_cur
    V_cur, S_cur_T = torch.linalg.qr(L_cur)

    # Return U, S (upper triangular), V
    return U_cur, S_cur_T.T, V_cur


def check_reduce_rank_equivalence(prev_matrix, rtol=1e-1, atol=1e-1):
    """Check equivalence between power series and SVD methods"""
    # Get SVD of previous matrix
    u, s, v = torch.linalg.svd(prev_matrix, full_matrices=False)

    cur_matrix = prev_matrix + torch.randn_like(prev_matrix)

    U_ps, S_ps, V_ps = reduce_rank_ps(cur_matrix, prev_matrix, u, torch.diag(s), v)
    reconst_ps = U_ps @ S_ps @ V_ps

    U_svd, S_svd, V_svd = torch.linalg.svd(cur_matrix, full_matrices=False)
    reconst_svd = U_svd[:, :3] @ torch.diag(S_svd[:3]) @ V_svd[:3, :]

    print("cur_matrix\n", cur_matrix)
    print("svd_matrix\n", reconst_svd)
    print("reconst_ps\n", reconst_ps)

    # Check equivalence
    assert torch.allclose(
        torch.abs(reconst_ps), torch.abs(reconst_svd), rtol=rtol, atol=atol
    ), f"Reconstruction mismatch: max diff = {torch.max(torch.abs(reconst_ps - cur_matrix))}"


class TestReduceRankPS:
    """Test class for power series rank reduction"""

    def setup_method(self):
        """Setup for each test method"""
        torch.manual_seed(42)

    def test_reduce_rank_equivalence_square(self):
        """Test equivalence for square matrices"""
        prev_matrix = torch.randn(5, 5)
        check_reduce_rank_equivalence(prev_matrix)


if __name__ == "__main__":
    # Run only this specific test
    pytest.main([__file__, "-v", "--tb=short"])
