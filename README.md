# AdaGram

Adaptive low-rank gradient preconditioning optimizers for PyTorch, with Adam variants and baseline methods for comparison.

## Install from branch

```bash
pip install "git+https://github.com/tnmtvv/AdaGram.git@develop"
```

Replace `develop` with any branch name to install from that branch.

## Usage

```python
import torch.nn as nn
from adagram_optimizers.AdagramPS import AdaGramPS

model = nn.Linear(10, 1)
optimizer = AdaGramPS(model.parameters(), lr=0.1, eps=1e-2, max_rank=8)
```

## Optimizers

### AdaGram family

| Optimizer | Description |
|-----------|-------------|
| `AdaGramPS` | AdaGram with projector-splitting rank updates. |
| `AdaGramFR` | AdaGram that compresses the low-rank factor via SVD when rank exceeds `max_rank`. |
| `AdaGramEQ` | AdaGramPS variant with keeping the singular values unite in the decomposition. |
| `SymAdaGram` | Symmetric projector-splitting AdaGram (see [arXiv:2010.02022](https://arxiv.org/pdf/2010.02022)). |
| `AdaGramPS_Sqrt` | Sqrt-preconditioned AdaGram with projector-splitting updates. |
| `AdaGramFR_Sqrt` | Sqrt-preconditioned AdaGram with SVD-based rank reduction. |

### Adam + AdaGram

| Optimizer | Description |
|-----------|-------------|
| `AdamGram` | `AdaGramPS` combined with Adam's first- and second-moment estimates. |
| `SymAdamGram` | `SymAdaGram` combined with Adam momentum. |
| `EQAdamGram` | `AdaGramEQ` combined with Adam momentum. |
| `SVDAdamGram` | `AdaGramFR` combined with Adam momentum. |
| `AdamGramSqrt_PS` | `AdaGramPS_Sqrt` combined with Adam momentum. |
| `AdamGramSqrt_SVD` | `AdaGramFR_Sqrt` combined with Adam momentum. |

### Baselines

| Optimizer | Description |
|-----------|-------------|
| `CustomAdaGrad` | Standard diagonal (element-wise) AdaGrad preconditioning. |
| `FullAdaGrad` | Full-matrix AdaGrad using an accumulated Gram matrix and eigendecomposition. |
| `Shampoo` | Matrix-valued preconditioner for 2D parameters with periodic Kronecker-factor updates. |
| `KATE` | Adaptive optimizer from the [KATE](https://github.com/nazya/KATE) implementation. |

## Requirements

Python ≥ 3.11, PyTorch ≥ 2.5. See `pyproject.toml` or `requirements.txt` for the full dependency list.
