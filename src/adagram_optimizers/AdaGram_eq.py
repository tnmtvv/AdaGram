import torch

from typing import Optional, Dict, Any, Tuple

from AdagramBase import AdaGram, AdaGramLogger
from AdagramPS import AdaGramPS

class AdaGramEQ(AdaGramPS):
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


    def one_rank_psi(self, b, g, u, s, v):
        # Ensure all inputs are 1D and on the same device/dtype
        g = g.flatten()
        u = u.flatten()
        # s = s.flatten()
        s = torch.tensor(1.0)
        v = v.flatten()

        if self.alpha:
            alpha = self.alpha 
        else: 
            alpha = 1 

        # Precompute scalar products
        gu = torch.dot(g, s * u)       # P = u*s
        gv = torch.dot(g, v)           # g^T v
        vv = torch.dot(v, v)           # v^T v

        # Only one vector multiplication per use
        const = b * (gv - gu * vv)     # Scalar
        delta_av = const * g           # Vector

        # Compute K and norm efficiently
        K_cur = u * s + delta_av       # Vector
        K_norm = torch.sqrt(torch.dot(K_cur, K_cur))     # Scalar

        U_cur = K_cur / K_norm         # Unit vector

        # S_hat = K_norm, S_tild = S_hat - <U_cur, delta_av>
        S_tild = K_norm - torch.dot(U_cur, delta_av)  # Scalar

        gk = torch.dot(g, U_cur)                      # Scalar

        # delta_au is fully vectorized, reuses gu, gk
        delta_au = b * (g - v * gu) * gk              # Vector

        # L_cur etc
        L_cur = v * S_tild + delta_au                 # Vector
        L_norm = torch.sqrt(torch.dot(L_cur, L_cur))                    # Scalar
        V_cur = L_cur / L_norm                        # Unit vector

        # Return as column vectors if needed
        return U_cur.unsqueeze(1), L_norm, V_cur.unsqueeze(1)

    def reduce_rank_psi(self, delta_A, U_0, S_0, V_0):
        if self.alpha:
            S_0 = self.alpha * S_0
            delta_A = (2 - self.alpha) * delta_A
            
            print("alpha", self.alpha)
        K_cur = U_0 @ S_0 + delta_A @ V_0
        U_cur, S_hat = torch.linalg.qr(K_cur)
        S_tild = S_hat - U_cur.T @ (delta_A @ V_0)
        L_cur = V_0 @ S_tild.T + delta_A.T @ U_cur
        V_cur, S_cur_T = torch.linalg.qr(L_cur)

        U, _, Vh = torch.linalg.svd(S_cur_T.T, full_matrices = False)
        
        new_U = U_cur @ U
        new_V = Vh @ V_cur
        new_S = torch.eye(S_cur_T.shape)

        return new_U, new_S, new_V