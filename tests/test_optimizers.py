import pytest
import torch
import torch.nn as nn
from typing import Dict, List, Any, Type
import json
import os
from torch.utils.data import TensorDataset, DataLoader

import libcontext

from src.adagram_optimizers.AdagramSVD import AdaGramFR
from src.adagram_optimizers.AdagramVanilla import AdaGramVanilla
from src.adagram_optimizers.AdagramPS import AdaGramPS
from src.adagram_optimizers.Shampoo import Shampoo
from src.adagram_optimizers.FullAdagrad import FullAdaGrad
from src.adagram_optimizers.AdamGram import AdamGram
from src.adagram_optimizers.AdaGram_eq import AdaGramEQ
from src.adagram_optimizers.SymAdaGram import SymAdaGram
from src.adagram_optimizers.AdamGram import SymAdamGram
from src.adagram_optimizers.AdamGram import SVDAdamGram
from src.adagram_optimizers.AdamGram import EQAdamGram


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


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def simple_model():
    return nn.Linear(10, 1)


@pytest.fixture
def complex_model():
    return nn.Sequential(
        nn.Linear(20, 50), nn.ReLU(), nn.Linear(50, 10), nn.ReLU(), nn.Linear(10, 1)
    )


@pytest.fixture
def sample_data():
    torch.manual_seed(42)
    return {
        "simple": (torch.randn(32, 10), torch.randn(32, 1)),
        "complex": (torch.randn(32, 20), torch.randn(32, 1)),
    }


@pytest.fixture
def loss_function():
    return nn.MSELoss()


@pytest.fixture
def optimizer_configs():
    return {
        "AdaGram":      {"lr": 0.1, "eps": 1e-4, "enable_logging": True},
        "AdaGramFR":    {"lr": 0.1, "eps": 1e-2, "max_rank": None, "enable_logging": True},
        "AdaGramPS":    {"lr": 0.1, "eps": 1e-2, "max_rank": None, "enable_logging": True},
        "AdamGram":     {"lr": 0.1, "eps": 1e-2, "max_rank": None, "enable_logging": True},
        "AdaGramEQ":    {"lr": 0.1, "eps": 1e-2, "max_rank": None, "enable_logging": True},
        "SymAdaGram":   {"lr": 0.1, "eps": 1e-2, "max_rank": None, "enable_logging": True},
        "SymAdamGram":  {"lr": 0.1, "eps": 1e-2, "max_rank": None, "enable_logging": True},
        "SVDAdamGram":  {"lr": 0.1, "eps": 1e-2, "max_rank": None, "enable_logging": True},
        "EQAdamGram":   {"lr": 0.1, "eps": 1e-2, "max_rank": None, "enable_logging": True},
    }


# ── Test class ────────────────────────────────────────────────────────────────

class TestOptimizerStateCapture:

    def _get_optimizer_and_atol(self, optimizer_name, model_params, configs):
        config = configs[optimizer_name]
        match optimizer_name:
            case "FullAdaGrad":
                return FullAdaGrad(model_params, **config), 1e-4
            case "AdaGram":
                return AdaGramVanilla(model_params, **config), 1e-4
            case "AdaGramFR":
                return AdaGramFR(model_params, **config), 1e-2
            case "AdaGramPS":
                return AdaGramPS(model_params, **config), 1e-2
            case "AdamGram":
                return AdamGram(model_params, **config), 1e-2
            case "AdaGramEQ":
                return AdaGramEQ(model_params, **config), 1e-2
            case "SymAdaGram":
                return SymAdaGram(model_params, **config), 1e-2
            case "SymAdamGram":
                return SymAdamGram(model_params, **config), 1e-2
            case "SVDAdamGram":
                return SVDAdamGram(model_params, **config), 1e-2
            case "EQAdamGram":
                return EQAdamGram(model_params, **config), 1e-2
            case _:
                raise ValueError(f"Unknown optimizer: {optimizer_name}")

    # ── test_state_initialization ─────────────────────────────────────────────

    @pytest.mark.parametrize(
        "optimizer_name",
        [
            "AdaGram",
            "AdaGramPS",
            "AdaGramFR",
            "AdamGram",
            "AdaGramEQ",
            "SymAdaGram",
            "SymAdamGram",
            "SVDAdamGram",
            "EQAdamGram",
        ],
    )
    def test_state_initialization(self, simple_model, optimizer_configs, optimizer_name):
        """Test that optimizer states are properly initialized"""
        config = optimizer_configs[optimizer_name]

        optimizer, _ = self._get_optimizer_and_atol(
            optimizer_name, simple_model.parameters(), optimizer_configs
        )
        tester = OptimizerStateTester(optimizer, optimizer_name)

        assert len(tester.states_history) == 0

        dummy_input = torch.randn(1, 10)
        dummy_target = torch.randn(1, 1)
        loss = nn.MSELoss()(simple_model(dummy_input), dummy_target)
        loss.backward()

        tester.capture_state()
        assert len(tester.states_history) == 1

        optimizer.step()
        tester.capture_state()
        assert len(tester.states_history) == 2

    # ── test_state_properties_minibatch ──────────────────────────────────────

    @pytest.mark.parametrize(
        "optimizer_name",
        [
            "AdaGramFR",
            "AdaGramPS",
            "SymAdaGram",
        ],
    )
    def test_state_properties_minibatch(
        self, simple_model, sample_data, loss_function, optimizer_configs, optimizer_name
    ):
        epochs = 5
        batch_size = 8
        model = simple_model

        X, y = sample_data["simple"]
        dataloader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=True)

        optimizer, atol = self._get_optimizer_and_atol(
            optimizer_name, model.parameters(), optimizer_configs
        )
        tester = OptimizerStateTester(optimizer, optimizer_name)

        for epoch in range(epochs):
            for batch_data, batch_targets in dataloader:
                optimizer.zero_grad()
                loss = loss_function(model(batch_data), batch_targets)
                loss.backward()
                tester.capture_state()
                optimizer.step()
                tester.loss_history.append(loss.item())

        assert len(tester.states_history) > 0
        for history in tester.states_history:
            for param_id, state_data in history["states"].items():
                optimizer_state = state_data["optimizer_state"]
                print("true_steps_num", optimizer_state["step_count"])

                assert "G" in optimizer_state, "G matrix should exist in state"
                assert "L_t" in optimizer_state, "L_t should exist in state"

                G_matrix = optimizer_state["G"]
                L_t = optimizer_state["L_t"]

                assert G_matrix.shape[0] == G_matrix.shape[1], "G matrix should be square"
                reconstructed_G = L_t @ L_t.T
                assert torch.allclose(G_matrix, reconstructed_G, atol=atol, rtol=atol)

    # ── test_reconstruction_minibatch ─────────────────────────────────────────

    @pytest.mark.parametrize(
        "optimizer_name",
        [
            "AdaGramFR",
            "AdaGramPS",
            "SymAdaGram",
            "SymAdamGram",
        ],
    )
    def test_reconstruction_minibatch(
        self, simple_model, sample_data, loss_function, optimizer_configs, optimizer_name
    ):
        epochs = 5
        batch_size = 8
        model = simple_model

        X, y = sample_data["simple"]
        dataloader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=True)

        optimizer, atol = self._get_optimizer_and_atol(
            optimizer_name, model.parameters(), optimizer_configs
        )
        max_rank = optimizer_configs[optimizer_name]["max_rank"]
        tester = OptimizerStateTester(optimizer, optimizer_name)

        for epoch in range(epochs):
            for batch_data, batch_targets in dataloader:
                optimizer.zero_grad()
                loss = loss_function(model(batch_data), batch_targets)
                loss.backward()
                tester.capture_state()
                optimizer.step()
                tester.loss_history.append(loss.item())

        assert len(tester.states_history) > 0
        for history in tester.states_history:
            for param_id, state_data in history["states"].items():
                optimizer_state = state_data["optimizer_state"]
                if "U" in optimizer_state:
                    rec_target = optimizer_state["rec_target"]
                    U, S, V = optimizer_state["U"], optimizer_state["S"], optimizer_state["V"]

                    factorize = S * U @ V.T if max_rank == 1 else U @ S @ V.T
                    assert torch.allclose(rec_target, factorize, atol=atol, rtol=atol)