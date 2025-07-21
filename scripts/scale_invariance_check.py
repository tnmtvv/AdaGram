#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import torch
import libcontext
from typing import Optional

from src.adagram_fixed_rank import AdaGramFR
from src.adagram_vanilla import AdaGramVanilla
from src.adagram_projector_splitting import AdaGramPS

n_feature = 20  # dimension of data
n_data = 100  # number of data-points
n_iteration = int(1e5)
n_trial = 1
n_epoch = 10
S = 1  # size of mini batch

### Initialize the data matrix and optimal weight
np.random.seed(0)
X = np.random.normal(0, 1, (n_data, n_feature)) / np.sqrt(n_feature)
w_optimal = np.random.normal(0, 10, n_feature)

### Generate the scaling diagonal matrix V such that each of the elements is e^r where r is generated from Uniform(-1, 1)
V = np.diag(np.exp(np.random.uniform(-1, 1, n_feature)))
V_inv_sq = np.linalg.matrix_power(np.linalg.inv(V), 2)
### Scale the data matrix using V
X_scaled = X @ V

### Fix the array of labels y
y = np.sign(X_scaled @ w_optimal)
y[y == 0] = (
    1  # Note that np.sign(0) = 0. So we set the label y = 1 in case any of the y is 0.
)

### Define Functions


### Define F
def F(X, y, w, lmd1=0):
    r = -y * X.dot(w)
    expr = np.exp(r)
    return np.mean(np.log(1 + expr)) + 0.5 * lmd1 * np.linalg.norm(w) ** 2


### Define Grad F
def dF(X, y, w, lmd1=0):
    r = -y * X.dot(w)
    expr = np.exp(r)
    grad = X.T.dot(-expr / (1 + expr) * y)
    return grad + lmd1 * w


### CRITICAL: Define shared initial weight vector
# This ensures both algorithms start from exactly the same point
SHARED_INITIAL_W = np.zeros(n_feature)  # Start from zero vector


def optimize_pytorch(X, y, optimizer_class, lr=0.01, device="cpu", initial_w=None):
    X_torch = torch.tensor(X, dtype=torch.float32, device=device)
    y_torch = torch.tensor(y, dtype=torch.float32, device=device)

    # Initialize parameters with shared initial weight
    if initial_w is not None:
        w = torch.tensor(
            initial_w.copy(), requires_grad=True, device=device, dtype=torch.float32
        )
    else:
        w = torch.zeros(X.shape[1], requires_grad=True, device=device)

    optimizer = optimizer_class([w], lr=lr, eps=1e-1, max_rank=5)

    history = []

    # Store the initial loss value for verification
    with torch.no_grad():
        w_np = w.detach().cpu().numpy()
        initial_loss = F(X, y, w_np)
        initial_grad_norm = np.linalg.norm(dF(X, y, w_np) / n_data) ** 2
        initial_accuracy = np.sum(y * (X @ w_np) >= 0) / n_data
        history.append((initial_loss, initial_grad_norm, initial_accuracy))

    for epo in range(n_epoch):
        for it in range(n_data):
            i = np.random.randint(n_data)
            j = min(n_data, i + S)

            X_batch = X_torch[i:j]
            y_batch = y_torch[i:j]

            # Compute loss
            r = -y_batch * (X_batch @ w)
            loss = torch.mean(torch.log(1 + torch.exp(r)))

            # Compute gradients
            optimizer.zero_grad()
            loss.backward()

            # Optimization step
            optimizer.step()

            # Record history
            with torch.no_grad():
                w_np = w.detach().cpu().numpy()
                if "scaled" in str(type(optimizer)).lower():
                    # For scaled data
                    full_loss = F(X, y, w_np)
                    grad_norm = (dF(X, y, w_np) / n_data).dot(
                        V_inv_sq @ (dF(X, y, w_np) / n_data)
                    )
                    accuracy = np.sum(y * (X @ w_np) >= 0) / n_data
                else:
                    # For original data
                    full_loss = F(X, y, w_np)
                    grad_norm = np.linalg.norm(dF(X, y, w_np) / n_data) ** 2
                    accuracy = np.sum(y * (X @ w_np) >= 0) / n_data

                history.append((full_loss, grad_norm, accuracy))

    return history


### Choose hyperparameters
beta = 1e-2
eta = 1 / ((dF(X, y, np.zeros(n_feature)) / n_data) ** 2)
etaScaled = 1 / ((dF(X_scaled, y, np.zeros(n_feature)) / n_data) ** 2)

### Initialize with shared starting point
w = SHARED_INITIAL_W.copy()  # Use shared initial point
wScaled = SHARED_INITIAL_W.copy()  # Use shared initial point

gScaled_sum = 0
gScaled_normsum = 0
g_sum = 0
g_normsum = 0

KATEhist = []
KATEhistScaled = []

# Store initial loss values for KATE algorithm
initial_loss = F(X, y, w)
initial_loss_scaled = F(X_scaled, y, wScaled)
initial_grad_norm = np.linalg.norm(dF(X, y, w) / n_data) ** 2
initial_grad_norm_scaled = (dF(X_scaled, y, wScaled) / n_data).dot(
    V_inv_sq @ (dF(X_scaled, y, wScaled) / n_data)
)
initial_accuracy = np.sum(y * X.dot(w) >= 0) / n_data
initial_accuracy_scaled = np.sum(y * X_scaled.dot(wScaled) >= 0) / n_data

KATEhist.append((initial_loss, initial_grad_norm, initial_accuracy))
KATEhistScaled.append(
    (initial_loss_scaled, initial_grad_norm_scaled, initial_accuracy_scaled)
)

### Run KATE optimizer
print("Running KATE optimizer...")
for epo in range(n_epoch):
    for it in range(n_data):
        i = np.random.randint(n_data)
        j = np.minimum(n_data, i + S)
        y_batch = y[i:j]  # batch label

        # for scaled data
        X_batch = X_scaled[i:j, :]  # batch Scaled data
        g = dF(X_batch, y_batch, wScaled) / S  # gradient of scaled data
        gScaled_sum += g * g
        gScaled_normsum += (g * g) / gScaled_sum
        wScaled -= (beta * np.sqrt(etaScaled * gScaled_sum + gScaled_normsum)) * (
            g / gScaled_sum
        )
        KATEhistScaled.append(
            (
                F(X_scaled, y, wScaled),
                (dF(X_scaled, y, wScaled) / n_data).dot(
                    V_inv_sq @ (dF(X_scaled, y, wScaled) / n_data)
                ),
                np.sum([y * X_scaled.dot(wScaled) >= 0]) / n_data,
            )
        )

        X_batch = X[i:j, :]
        g = dF(X_batch, y_batch, w) / S
        g_sum += g * g
        g_normsum += (g * g) / g_sum
        w -= (beta * np.sqrt(eta * g_sum + g_normsum)) * (g / g_sum)
        KATEhist.append(
            (
                F(X, y, w),
                np.linalg.norm(dF(X, y, w) / n_data) ** 2,
                np.sum([y * X.dot(w) >= 0]) / n_data,
            )
        )

### Run custom PyTorch optimizer with same initial point
print("Running custom PyTorch optimizer...")
# Set random seed for reproducibility
torch.manual_seed(0)
np.random.seed(0)

CustomHistPyTorch = optimize_pytorch(
    X, y, AdaGramPS, lr=0.1, initial_w=SHARED_INITIAL_W
)
CustomHistPyTorchScaled = optimize_pytorch(
    X_scaled, y, AdaGramPS, lr=0.1, initial_w=SHARED_INITIAL_W
)

# Verification: Print initial loss values
print(f"\nInitial loss verification:")
print(f"KATE (original): {KATEhist[0][0]:.6f}")
print(f"KATE (scaled): {KATEhistScaled[0][0]:.6f}")
print(f"AdaGram (original): {CustomHistPyTorch[0][0]:.6f}")
print(f"AdaGram (scaled): {CustomHistPyTorchScaled[0][0]:.6f}")

### Make plots comparing all optimizers
marker = np.arange(
    0, n_epoch * n_data + 1, max(1, (n_epoch * n_data + 1) // 100), dtype="int"
)

# Function value comparison
plt.figure(figsize=(12, 8))
plt.plot(
    marker,
    [KATEhist[i][0] for i in marker if i < len(KATEhist)],
    color="b",
    label=r"KATE: Dataset $(x_i,y_i)$",
)
plt.plot(
    marker,
    [KATEhistScaled[i][0] for i in marker if i < len(KATEhistScaled)],
    color="orange",
    linestyle="dashed",
    label=r"KATE: Dataset $(Vx_i,y_i)$",
)
plt.plot(
    marker,
    [CustomHistPyTorch[i][0] for i in marker if i < len(CustomHistPyTorch)],
    color="red",
    linestyle="dotted",
    label=r"AdaGramPS: Dataset $(x_i,y_i)$",
)
plt.plot(
    marker,
    [CustomHistPyTorchScaled[i][0] for i in marker if i < len(CustomHistPyTorchScaled)],
    color="green",
    linestyle="dashdot",
    label=r"AdaGramPS: Dataset $(Vx_i,y_i)$",
)
plt.grid(True)
plt.ylabel(r"$f(w_t)$", fontsize=15)
plt.xlabel("iterations", fontsize=15)
plt.legend(fontsize=12)
plt.title("Function Value Comparison", fontsize=16)
plt.savefig("scale_functionval_comparison_same_start.pdf")

# Gradient norm comparison
plt.figure(figsize=(12, 8))
plt.plot(
    marker,
    [KATEhist[i][1] for i in marker if i < len(KATEhist)],
    color="b",
    label=r"KATE: $\Vert \nabla f(w_t) \Vert^2$; Dataset $(x_i,y_i)$",
)
plt.plot(
    marker,
    [KATEhistScaled[i][1] for i in marker if i < len(KATEhistScaled)],
    color="orange",
    linestyle="dashed",
    label=r"KATE: $\Vert \nabla f(w_t) \Vert^2_{V^{-2}}$; Dataset $(Vx_i,y_i)$",
)
plt.plot(
    marker,
    [CustomHistPyTorch[i][1] for i in marker if i < len(CustomHistPyTorch)],
    color="red",
    linestyle="dotted",
    label=r"AdaGramPS: $\Vert \nabla f(w_t) \Vert^2$; Dataset $(x_i,y_i)$",
)
plt.plot(
    marker,
    [CustomHistPyTorchScaled[i][1] for i in marker if i < len(CustomHistPyTorchScaled)],
    color="green",
    linestyle="dashdot",
    label=r"AdaGramPS: $\Vert \nabla f(w_t) \Vert^2_{V^{-2}}$; Dataset $(Vx_i,y_i)$",
)
plt.yscale("log")
plt.grid(True)
plt.ylabel("Grad Norm", fontsize=15)
plt.xlabel("iterations", fontsize=15)
plt.legend(fontsize=10)
plt.title("Gradient Norm Comparison", fontsize=16)
plt.savefig("scale_grad_comparison_same_start.pdf")

# Accuracy comparison
plt.figure(figsize=(12, 8))
plt.plot(
    marker,
    [KATEhist[i][2] for i in marker if i < len(KATEhist)],
    color="b",
    label=r"KATE: Dataset $(x_i,y_i)$",
)
plt.plot(
    marker,
    [KATEhistScaled[i][2] for i in marker if i < len(KATEhistScaled)],
    color="orange",
    linestyle="dashed",
    label=r"KATE: Dataset $(Vx_i,y_i)$",
)
plt.plot(
    marker,
    [CustomHistPyTorch[i][2] for i in marker if i < len(CustomHistPyTorch)],
    color="red",
    linestyle="dotted",
    label=r"AdaGramPS: Dataset $(x_i,y_i)$",
)
plt.plot(
    marker,
    [CustomHistPyTorchScaled[i][2] for i in marker if i < len(CustomHistPyTorchScaled)],
    color="green",
    linestyle="dashdot",
    label=r"AdaGramPS: Dataset $(Vx_i,y_i)$",
)
plt.grid(True)
plt.ylabel("Accuracy", fontsize=15)
plt.xlabel("iterations", fontsize=15)
plt.legend(fontsize=12)
plt.title("Accuracy Comparison", fontsize=16)
plt.savefig("scale_accuracy_comparison_same_start.pdf")

plt.show()

print("Scale invariance testing complete!")
print(
    f"KATE optimizer final accuracies: Original={KATEhist[-1][2]:.4f}, Scaled={KATEhistScaled[-1][2]:.4f}"
)
print(
    f"AdaGram optimizer final accuracies: Original={CustomHistPyTorch[-1][2]:.4f}, Scaled={CustomHistPyTorchScaled[-1][2]:.4f}"
)
