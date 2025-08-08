import pytest
import torch
import torch.nn as nn
from typing import Dict, List, Any, Type
import json
import os

import libcontext


from src.AdagramSVD import AdaGramFR
from src.AdagramVanilla import AdaGramVanilla
from src.AdagramPS import AdaGramPS
from src.Shampoo import Shampoo
from src.FullAdagrad import FullAdaGrad


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
        "AdaGram": {"lr": 0.1, "eps": 1e-2},
        "AdaGramFR": {"lr": 0.1, "eps": 1e-2, "max_rank": None},
        "AdaGramPS": {"lr": 0.1, "eps": 1e-2, "max_rank": None},
    }


class TestOptimizerStateCapture:

    @pytest.mark.parametrize(
        "optimizer_name",
        [
            "AdaGram",
            "AdaGramPS",
            "AdaGramFR",
        ],
    )
    def test_state_initialization(
        self, simple_model, optimizer_configs, optimizer_name
    ):
        """Test that optimizer states are properly initialized"""
        config = optimizer_configs[optimizer_name]

        if optimizer_name == "AdaGram":
            optimizer = AdaGramVanilla(simple_model.parameters(), **config)
        elif optimizer_name == "AdaGramFR":
            optimizer = AdaGramFR(simple_model.parameters(), **config)
        elif optimizer_name == "AdaGramPS":
            optimizer = AdaGramPS(simple_model.parameters(), **config)

        tester = OptimizerStateTester(optimizer, optimizer_name)

        # Before any steps, state should be empty or have default values
        assert len(tester.states_history) == 0

        # Create gradient
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
            "AdaGram",
            "AdaGramFR",
            "AdaGramPS",
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
            atol = 1e-4
        elif optimizer_name == "AdaGram":
            optimizer = AdaGramVanilla(simple_model.parameters(), **config)
            atol = 1e-4
        elif optimizer_name == "AdaGramFR":
            optimizer = AdaGramFR(simple_model.parameters(), **config)
            atol = 1e-3
        elif optimizer_name == "AdaGramPS":
            optimizer = AdaGramPS(simple_model.parameters(), **config)
            atol = 1e-3

        tester = OptimizerStateTester(optimizer, optimizer_name)

        # Run a few optimization steps
        for step in range(10):
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

                reconstructed_G = L_t @ L_t.T
                print("reconstructed_G", reconstructed_G)
                print("G_matrix", G_matrix)

                error_norm = torch.norm(
                    torch.abs(G_matrix - reconstructed_G)
                ) / torch.norm(G_matrix)
                print("error_norm", error_norm)
                assert torch.allclose(G_matrix, reconstructed_G, atol=atol, rtol=atol)

    @pytest.mark.parametrize(
        "optimizer_name",
        ["AdaGramPS", "AdaGramFR"],
    )
    def test_reconstruction(
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

        if optimizer_name == "AdaGramFR":
            optimizer = AdaGramFR(simple_model.parameters(), **config)
            atol = 1e-3
        elif optimizer_name == "AdaGramPS":
            optimizer = AdaGramPS(simple_model.parameters(), **config)
            atol = 1e-3

        tester = OptimizerStateTester(optimizer, optimizer_name)

        # Run a few optimization steps
        for step in range(20):
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

                assert "G" in optimizer_state, "G matrix should exist in state"
                assert "L_t" in optimizer_state, "L_t should exist in state"

                if "U" in optimizer_state:
                    rec_target = optimizer_state["rec_target"]
                    rec_target = optimizer_state["P"] @ optimizer_state["Q"].T
                    U, S, V = (
                        optimizer_state["U"],
                        optimizer_state["S"],
                        optimizer_state["V"],
                    )
                    if optimizer_name == "AdaGramFR":
                        V = V.T

                    factorize = U @ S @ V.T

                    print("rec_target", rec_target)
                    print("factorize", factorize)

                    assert torch.allclose(rec_target, factorize, atol=atol, rtol=atol)


class TestOptimizerComparison:

    def setup_optimizers(self, model, optimizer_configs):
        """Setup multiple optimizers for comparison"""
        optimizers = {}
        testers = {}

        for name, config in optimizer_configs.items():
            # Create separate model instances to avoid interference
            test_model = type(model)(*[p for p in model.parameters()])
            test_model.load_state_dict(model.state_dict())

            if name == "FullAdaGrad":
                opt = FullAdaGrad(test_model.parameters(), **config)
            elif name == "AdaGram":
                opt = AdaGramVanilla(simple_model.parameters(), **config)

            optimizers[name] = {"model": test_model, "optimizer": opt}
            testers[name] = OptimizerStateTester(opt, name)

        return optimizers, testers
