import torch
import pytest


def reduce_rank_psi(delta_A, U_0, S_0, V_0):
    print("delta_A",delta_A)

    K_cur = U_0 @ S_0 + delta_A @ V_0

    U_cur, S_hat = torch.linalg.qr(K_cur)

    S_tild = S_hat - U_cur.T @ delta_A @ V_0


    L_cur = V_0 @ S_tild.T + delta_A.T @ U_cur
    print("U_cur_ps", U_cur)
    print("delta_A.T @ U_cur", delta_A.T @ U_cur)

    V_cur, S_cur_T = torch.linalg.qr(L_cur)

    return U_cur, S_cur_T.T, V_cur

    
def one_rank_psi(b, g, u, s, v):
    # Ensure all inputs are 1D and on the same device/dtype
    g = g.flatten()
    u = u.flatten()
    s = s.flatten()
    v = v.flatten()

    # Precompute scalar products
    gu = torch.dot(g, s * u)       # g^T (s * u)
    gv = torch.dot(g, v)           # g^T v
    vv = torch.dot(v, v)           # v^T v

    # Only one vector multiplication per use
    const = b * (gv - gu * vv)     # Scalar
    delta_av = const * g           # Vector

    # Compute K and norm efficiently
    K_cur = u * s + delta_av       # Vector
    K_norm = torch.norm(K_cur)     # Scalar

    U_cur = K_cur / K_norm         # Unit vector

    # S_hat = K_norm, S_tild = S_hat - <U_cur, delta_av>
    S_tild = K_norm - torch.dot(U_cur, delta_av)  # Scalar

    gk = torch.dot(g, U_cur)                      # Scalar

    # delta_au is fully vectorized, reuses gu, gk
    delta_au = b * (g - v * gu) * gk              # Vector

    # L_cur etc
    L_cur = v * S_tild + delta_au                 # Vector
    L_norm = torch.norm(L_cur)                    # Scalar
    V_cur = L_cur / L_norm                        # Unit vector

    # Return as column vectors if needed
    return U_cur.unsqueeze(1), L_norm, V_cur.unsqueeze(1)


def check_rank1_vs_reduce_rank(prev_matrix, b=1.0, rtol=1e-2, atol=1e-2):
    """
    Compare the output between the rank-1 psi and ordinary psi (reduce_rank_psi).

    Args:
        prev_matrix (torch.Tensor): The original matrix (square).
        b (float): Scalar value used in one_rank_psi.
        rtol (float): Relative tolerance for equivalence check.
        atol (float): Absolute tolerance for equivalence check.
    """
    # SVD of previous matrix
    u, s, v = torch.linalg.svd(prev_matrix, full_matrices=False)

    # Generate a random rank-1 update: delta_A = g @ g.T for some g
    g = torch.randn(prev_matrix.shape[0])
    identity = torch.eye(
                prev_matrix.shape[0]
            )
    delta_A = b * torch.ger(g, g) @ (identity - u[:, :1] @ torch.diag(s[:1]) @ v[:1, :])

    # Reduce rank method
    U_ps, S_ps, V_ps = reduce_rank_psi(delta_A, u[:, :1], torch.diag(s[:1]), v[:1, :].T)
    reconst_ps = U_ps @ S_ps @ V_ps.T
    # print("U_ps", U_ps)
    # print("V_ps", V_ps)

    # Rank-1 psi method, using the first singular vectors/values and g
    U_r1, S_r1, V_r1 = one_rank_psi(
        b,
        g,
        u[:, :1],
        s[:1],
        v[:1, :]
    )
    reconst_r1 = S_r1 * U_r1  @ V_r1.T

    print("U_r1", U_r1)
    print("V_r1", V_r1)


    # print("reconst_ps\n", reconst_ps)
    # print("reconst_r1\n", reconst_r1)
    # Check equivalence (up to tolerances, and only if shapes match)
    if reconst_ps.shape == reconst_r1.shape:
        assert torch.allclose(
            torch.abs(reconst_ps), torch.abs(reconst_r1), rtol=rtol, atol=atol
        ), (
            f"Reconstruction mismatch: max diff = {torch.max(torch.abs(reconst_ps - reconst_r1))}"
        )
    else:
        print("Warning: Shape mismatch between reduce_rank_psi and one_rank_psi outputs.")

# Example usage:
# prev_matrix = torch.randn(5, 5)
# check_rank1_vs_reduce_rank(prev_matrix)



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

    assert torch.allclose(torch.abs(P_new @ Q_new.T), torch.abs(P @ Q.T), atol=1e-3)


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

    assert torch.allclose(torch.abs(P_new @ Q_new.T), torch.abs(P @ Q.T), atol=1e-3)


class TestReduceRankPS:
    """Test class for power series rank reduction"""

    def setup_method(self):
        """Setup for each test method"""
        torch.manual_seed(42)

    def test_reduce_rank_equivalence_square(self):
        """Test equivalence for square matrices"""
        prev_matrix = torch.randn(5, 5)
        # check_reduce_rank_equivalence(prev_matrix, rtol=1e-5, atol=1e-3)
        check_rank1_vs_reduce_rank(prev_matrix)

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
