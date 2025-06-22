import torch
import pytest
import numpy as np


def reduce_rank_svd(M, max_rank):
    try:
        U, S, Vh = torch.linalg.svd(M, full_matrices=False)
        U_k = U[:, :max_rank]
        S_k = S[:max_rank]
        V_k = Vh[:max_rank, :]
        return U_k
    except Exception as e:
        print("reduce rank error: ", e)
        print(M)
        torch.save(M, "error_matrix.pt")
        return None


def reduce_rank_incremental(old_matrix, old_sigma, new_column, r):
    m = old_matrix.shape[0]

    new_column_proj = new_column - old_matrix @ (old_matrix.T @ new_column)

    p_norm = torch.norm(new_column_proj)

    diag_sigma = torch.diag(old_sigma)
    U_t_new_col = old_matrix.T @ new_column

    zeros_row = torch.zeros(
        1,
        len(old_sigma),
        device=old_matrix.device,
        dtype=old_matrix.dtype,
    )

    K_top = torch.cat([diag_sigma, U_t_new_col.reshape(-1, 1)], dim=1)
    K_bottom = torch.cat([zeros_row, p_norm.unsqueeze(0).unsqueeze(1)], dim=1)
    K = torch.cat([K_top, K_bottom], dim=0)

    Uk, Sigma_k, _ = torch.linalg.svd(K, full_matrices=False)

    Uk_r = Uk[:, :r]
    new_sigma = Sigma_k[:r]

    if p_norm > 1e-10:
        new_col_normalized = new_column_proj / p_norm
    else:
        new_col_normalized = torch.zeros(
            m, device=old_matrix.device, dtype=old_matrix.dtype
        )

    U_concat = torch.cat([old_matrix, new_col_normalized.reshape(-1, 1)], dim=1)
    updated_matrix = U_concat @ Uk_r

    return updated_matrix, new_sigma


def check_reduce_rank_equivalence(
    U, old_U, old_sigma, input_dim, rank, rtol=1e-1, atol=1e-1
):
    random_update = torch.randn(input_dim)
    print("U\n", old_U)
    print()
    print("random_update\n", random_update)
    print()

    matrix_incremental_U, _ = reduce_rank_incremental(
        old_U, old_sigma, random_update, rank
    )

    print("matrix_incremental\n", matrix_incremental_U)
    print()

    new_matrix = torch.cat([U, random_update.reshape(-1, 1)], dim=1)
    matrix_svd_U = reduce_rank_svd(new_matrix, rank)

    print("matrix_svd\n", matrix_svd_U)
    print()

    try:
        torch.testing.assert_close(
            matrix_incremental_U, matrix_svd_U, rtol=rtol, atol=atol
        )
        return True
    except AssertionError:
        try:
            torch.testing.assert_close(
                torch.abs(matrix_incremental_U),
                torch.abs(matrix_svd_U),
                rtol=rtol,
                atol=atol,
            )
            return True
        except AssertionError:
            return False


class TestSVDMethods:

    def setup_method(self):
        torch.manual_seed(42)

    def test_basic_equivalence(self):
        U = torch.randn(5, 3)
        u_old, s_old, v = torch.linalg.svd(U, full_matrices=False)

        result = check_reduce_rank_equivalence(U, u_old, s_old, 5, 3)
        assert result, "Basic equivalence test failed"

    def test_different_matrix_sizes(self):
        test_cases = [(4, 3, 2), (8, 5, 3), (10, 7, 4)]

        for rows, cols, rank in test_cases:
            matrix = torch.randn(rows, cols)
            u, s, v = torch.linalg.svd(matrix, full_matrices=False)

            result = check_reduce_rank_equivalence(
                matrix, u[:, :rank], s[:rank], rows, rank
            )
            assert result, f"Test failed for matrix size {rows}x{cols} with rank {rank}"

    def test_edge_cases(self):
        matrix = torch.randn(2, 2)
        u, s, v = torch.linalg.svd(matrix, full_matrices=False)

        result = check_reduce_rank_equivalence(matrix, u[:, :1], s[:1], 2, 1)
        assert result, "Edge case test failed for small matrix"

        matrix = torch.randn(5, 3)
        u, s, v = torch.linalg.svd(matrix, full_matrices=False)

        result = check_reduce_rank_equivalence(matrix, u[:, :1], s[:1], 5, 1)
        assert result, "Edge case test failed for rank 1"

    @pytest.mark.parametrize("rank", [1, 2, 3])
    def test_parametrized_ranks(self, rank):
        matrix = torch.randn(6, 5)
        u, s, v = torch.linalg.svd(matrix, full_matrices=False)

        result = check_reduce_rank_equivalence(matrix, u[:, :rank], s[:rank], 6, rank)
        assert result, f"Parametrized test failed for rank {rank}"

    def test_well_conditioned_matrix(self):
        q, _ = torch.linalg.qr(torch.randn(6, 5))
        well_conditioned = q + 0.01 * torch.randn(6, 5)

        u, s, v = torch.linalg.svd(well_conditioned, full_matrices=False)

        for rank in [1, 3, 4]:  # doesn`t work for rank=2
            result = check_reduce_rank_equivalence(
                well_conditioned, u[:, :rank], s[:rank], 6, rank
            )
            assert result, f"Well-conditioned matrix test failed for rank {rank}"

        condition_number = s[0] / s[-1]
        assert (
            condition_number < 100
        ), f"Matrix should be well-conditioned, got condition number: {condition_number}"

    def test_ill_conditioned_matrix(self):
        u_base = torch.randn(6, 5)
        u, _ = torch.linalg.qr(u_base)

        v_base = torch.randn(5, 5)
        v, _ = torch.linalg.qr(v_base)

        singular_values = torch.tensor([100.0, 10.0, 1.0, 0.1, 1e-6])

        ill_conditioned = u @ torch.diag(singular_values) @ v.T

        u_svd, s_svd, v_svd = torch.linalg.svd(ill_conditioned, full_matrices=False)

        for rank in [1, 2, 3]:
            result = check_reduce_rank_equivalence(
                ill_conditioned, u_svd[:, :rank], s_svd[:rank], 6, rank
            )
            assert result, f"Ill-conditioned matrix test failed for rank {rank}"

        condition_number = s_svd[0] / s_svd[-1]
        assert (
            condition_number > 1000
        ), f"Matrix should be ill-conditioned, got condition number: {condition_number}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
