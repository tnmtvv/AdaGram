import pytest
import torch
import torch.nn as nn
from typing import Dict, List, Any, Type
import json
import os

import libcontext


from src.adagram_fixed_rank import AdaGramFR
from src.adagram_vanilla import AdaGramVanilla
from src.adagram_projector_splitting import AdaGramPS
from src.shampoo import Shampoo
from src.full_G import FullAdaGrad


class OptimizerStateTester:
    def __init__(self, optimizer, name: str):
        self.optimizer = optimizer
        self.name = name
        self.states_history = []
        self.step_count = 0
        self.loss_history = []

    def capture_state(self):
        """Capture current optimizer state"""
        state_snapshot = {}
        for group_idx, group in enumerate(self.optimizer.param_groups):
            for param_idx, param in enumerate(group["params"]):
                param_id = f"group_{group_idx}_param_{param_idx}"
                if param in self.optimizer.state:
                    state_snapshot[param_id] = {
                        "param_shape": list(param.shape),
                        "param_data": param.data.clone(),
                        "grad_data": (
                            param.grad.clone() if param.grad is not None else None
                        ),
                        "optimizer_state": self._deep_copy_state(
                            self.optimizer.state[param]
                        ),
                    }

        self.states_history.append({"step": self.step_count, "states": state_snapshot})
        self.step_count += 1

    def _deep_copy_state(self, state):
        """Deep copy optimizer state tensors"""
        copied_state = {}
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                copied_state[key] = value.clone()
            else:
                copied_state[key] = value
        return copied_state


# Pytest fixtures
@pytest.fixture
def simple_model():
    """Simple linear model for testing"""
    return nn.Linear(10, 1)


@pytest.fixture
def complex_model():
    """More complex model for testing"""
    return nn.Sequential(
        nn.Linear(20, 50), nn.ReLU(), nn.Linear(50, 10), nn.ReLU(), nn.Linear(10, 1)
    )


@pytest.fixture
def sample_data():
    """Sample training data"""
    torch.manual_seed(42)
    return {
        "simple": (torch.randn(32, 10), torch.randn(32, 1)),
        "complex": (torch.randn(32, 20), torch.randn(32, 1)),
    }


@pytest.fixture
def loss_function():
    """Loss function for testing"""
    return nn.MSELoss()


@pytest.fixture
def optimizer_configs():
    """Configuration for different optimizers"""
    return {
        "FullAdaGrad": {"lr": 0.1, "eps": 1e-5},
        "AdaGram": {"lr": 0.1, "eps": 1e-5},
        "AdaGramFR": {"lr": 0.1, "eps": 1e-5, "max_rank": 2},
        "AdaGramPS": {"lr": 0.1, "eps": 1e-5, "max_rank": 10},
    }


class TestOptimizerStateCapture:

    @pytest.mark.parametrize(
        "optimizer_name",
        [
            # "FullAdaGrad",
            "AdaGram",
            # "AdaGramPS",
            # "AdaGramFR"
        ],
    )
    def test_state_initialization(
        self, simple_model, optimizer_configs, optimizer_name
    ):
        """Test that optimizer states are properly initialized"""
        config = optimizer_configs[optimizer_name]

        if optimizer_name == "FullAdaGrad":
            optimizer = FullAdaGrad(simple_model.parameters(), **config)
        elif optimizer_name == "AdaGram":
            optimizer = AdaGramVanilla(simple_model.parameters(), **config)
        elif optimizer_name == "AdaGramFR":
            optimizer = AdaGramFR(simple_model.parameters(), **config)
        elif optimizer_name == "AdaGramPS":
            optimizer = AdaGramPS(simple_model.parameters(), **config)

        tester = OptimizerStateTester(optimizer, optimizer_name)

        # Before any steps, state should be empty or have default values
        assert len(tester.states_history) == 0

        # Create dummy gradient
        dummy_input = torch.randn(1, 10)
        dummy_target = torch.randn(1, 1)
        loss = nn.MSELoss()(simple_model(dummy_input), dummy_target)
        loss.backward()

        # Capture state after backward pass
        tester.capture_state()
        assert len(tester.states_history) == 1

        # Take optimizer step
        optimizer.step()
        tester.capture_state()
        assert len(tester.states_history) == 2

    @pytest.mark.parametrize(
        "optimizer_name",
        [
            # "FullAdaGrad",
            "AdaGram",
            # "AdaGramFR",
            # "AdaGramPS"
        ],
    )
    def test_state_properties(
        self,
        simple_model,
        sample_data,
        loss_function,
        optimizer_configs,
        optimizer_name,
    ):
        """Test specific properties of FullAdaGrad states"""
        model = simple_model
        data, targets = sample_data["simple"]

        """Test that optimizer states are properly initialized"""
        config = optimizer_configs[optimizer_name]

        if optimizer_name == "FullAdaGrad":
            optimizer = FullAdaGrad(simple_model.parameters(), **config)
        elif optimizer_name == "AdaGram":
            optimizer = AdaGramVanilla(simple_model.parameters(), **config)
        elif optimizer_name == "AdaGramFR":
            optimizer = AdaGramFR(simple_model.parameters(), **config)
        elif optimizer_name == "AdaGramPS":
            optimizer = AdaGramPS(simple_model.parameters(), **config)

        tester = OptimizerStateTester(optimizer, optimizer_name)

        # Run a few optimization steps
        for step in range(15):
            optimizer.zero_grad()
            output = model(data)
            loss = loss_function(output, targets)
            loss.backward()

            tester.capture_state()
            optimizer.step()
            tester.loss_history.append(loss.item())

        # Test properties
        for history in tester.states_history:
            for param_id, state_data in history["states"].items():
                optimizer_state = state_data["optimizer_state"]

                assert (
                    "G" in optimizer_state
                ), "G matrix should exist in FullAdaGrad state"
                assert "L_t" in optimizer_state, "L_t should exist in FullAdaGrad state"

                G_matrix = optimizer_state["G"]
                L_t = optimizer_state["L_t"]

                # G should be square
                assert (
                    G_matrix.shape[0] == G_matrix.shape[1]
                ), "G matrix should be square"

                # Check the relationship between L_t and G matrix
                # Note: Using torch.allclose with tolerance for numerical stability

                eigenvals, eigenvecs = torch.linalg.eigh(G_matrix)
                sqrt_eigenvals = torch.sqrt(eigenvals)
                sqr_G = eigenvecs @ torch.diag(sqrt_eigenvals) @ eigenvecs.T

                reconstructed_G = L_t @ L_t.T
                print("reconstructed_G", reconstructed_G)
                print("G_matrix", G_matrix)

                error_norm = torch.norm(
                    torch.abs(G_matrix - reconstructed_G)
                ) / torch.norm(G_matrix)
                print(error_norm)
                assert torch.allclose(sqr_G, L_t, atol=1e-5, rtol=1e-5)


# class TestOptimizerComparison:

#     def setup_optimizers(self, model, optimizer_configs):
#         """Setup multiple optimizers for comparison"""
#         optimizers = {}
#         testers = {}

#         for name, config in optimizer_configs.items():
#             # Create separate model instances to avoid interference
#             test_model = type(model)(*[p for p in model.parameters()])
#             test_model.load_state_dict(model.state_dict())

#             if name == "FullAdaGrad":
#                 opt = FullAdaGrad(test_model.parameters(), **config)
#             elif name == "AdaGram":
#                 opt = AdaGramVanilla(simple_model.parameters(), **config)

#             optimizers[name] = {"model": test_model, "optimizer": opt}
#             testers[name] = OptimizerStateTester(opt, name)

#         return optimizers, testers

#     def test_convergence_comparison(
#         self, simple_model, sample_data, loss_function, optimizer_configs
#     ):
#         """Compare convergence behavior across optimizers"""
#         data, targets = sample_data["simple"]
#         optimizers, testers = self.setup_optimizers(simple_model, optimizer_configs)

#         num_steps = 50

#         for step in range(num_steps):
#             for name, components in optimizers.items():
#                 model = components["model"]
#                 optimizer = components["optimizer"]
#                 tester = testers[name]

#                 optimizer.zero_grad()
#                 output = model(data)
#                 loss = loss_function(output, targets)
#                 loss.backward()

#                 tester.capture_state()
#                 tester.loss_history.append(loss.item())

#                 optimizer.step()

#         # Analyze convergence
#         for name, tester in testers.items():
#             # Check that loss generally decreases
#             initial_loss = tester.loss_history[0]
#             final_loss = tester.loss_history[-1]

#             # Allow for some fluctuation but expect overall improvement
#             assert final_loss < initial_loss * 1.1, f"{name} did not converge properly"

#             # Check that we have captured states for all steps
#             assert len(tester.states_history) == num_steps


# @pytest.mark.parametrize(
#     "optimizer_name", ["FullAdaGrad", "AdaGram", "AdaGramFR", "AdaGramPS"]
# )
# def test_state_matrix_frobenius_norm(
#     self, simple_model, sample_data, loss_function, optimizer_configs, optimizer_name
# ):
#     """Test Frobenius norm and validity of P and Q matrices for different optimizers"""
#     data, targets = sample_data["simple"]

#     # Initialize optimizer based on name
#     config = optimizer_configs[optimizer_name]
#     if optimizer_name == "FullAdaGrad":
#         optimizer = FullAdaGrad(simple_model.parameters(), **config)
#     elif optimizer_name == "AdaGram":
#         optimizer = AdaGramVanilla(simple_model.parameters(), **config)
#     elif optimizer_name == "AdaGramFR":
#         optimizer = AdaGramFR(simple_model.parameters(), **config)
#     elif optimizer_name == "AdaGramPS":
#         optimizer = AdaGramPS(simple_model.parameters(), **config)
#     else:
#         raise ValueError(f"Unknown optimizer: {optimizer_name}")

#     tester = OptimizerStateTester(optimizer, optimizer_name)

#     # Run optimization steps
#     for step in range(10):
#         optimizer.zero_grad()
#         output = simple_model(data)
#         loss = loss_function(output, targets)
#         loss.backward()

#         tester.capture_state()
#         optimizer.step()

#     # Analyze matrix properties
#     for history in tester.states_history:
#         for param_id, state_data in history["states"].items():
#             P_matrix = state_data["optimizer_state"]["P"]
#             Q_matrix = state_data["optimizer_state"]["Q"]

#             M = P_matrix @ Q_matrix.T

#             # Check for NaN and infinity values
#             assert torch.isfinite(
#                 M
#             ).any(), f"P matrix for param {param_id} contains NaN or Inf"
