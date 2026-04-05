"""Tests for ARMFeedForward (ARM-FFN) routing."""

import torch
import pytest
from dstt import DSTTConfig, ARMFeedForward


@pytest.fixture
def config():
    return DSTTConfig.tiny()


class TestARMFeedForward:

    def test_output_shape(self, config):
        ffn = ARMFeedForward(config)
        x = torch.randn(2, 32, config.d_model)
        out = ffn(x)
        assert out.shape == (2, 32, config.d_model)

    def test_load_balance_loss_computed(self, config):
        ffn = ARMFeedForward(config)
        x = torch.randn(2, 32, config.d_model)
        _ = ffn(x)
        lb = ffn.load_balance_loss
        assert lb.item() >= 0

    def test_expert_count(self, config):
        ffn = ARMFeedForward(config)
        assert len(ffn.experts) == config.n_experts

    def test_gradient_flow(self, config):
        ffn = ARMFeedForward(config)
        x = torch.randn(2, 16, config.d_model, requires_grad=True)
        out = ffn(x)
        out.sum().backward()
        assert x.grad is not None

    def test_centroid_shape(self, config):
        ffn = ARMFeedForward(config)
        assert ffn.centroids.shape == (config.n_experts, config.d_model)
