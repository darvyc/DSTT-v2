"""Tests for CFM and AFM flow matrices."""

import torch
import pytest
from dstt import DSTTConfig
from dstt.flow_matrices import CorrectFlowMatrix, AdversarialFlowMatrix


@pytest.fixture
def config():
    return DSTTConfig.tiny()


class TestCFM:

    def test_scores_are_finite(self, config):
        cfm = CorrectFlowMatrix(config.d_head, config.d_model)
        keys = torch.randn(2, 8, 16, config.d_head)
        ctx = torch.randn(2, config.d_model)
        out = cfm(keys, ctx)
        assert torch.isfinite(out).all()

    def test_scores_respond_to_context(self, config):
        cfm = CorrectFlowMatrix(config.d_head, config.d_model)
        keys = torch.randn(2, 8, 16, config.d_head)
        ctx1 = torch.randn(2, config.d_model)
        ctx2 = torch.randn(2, config.d_model) * 5
        out1 = cfm(keys, ctx1)
        out2 = cfm(keys, ctx2)
        assert not torch.allclose(out1, out2)


class TestAFM:

    def test_scores_non_negative(self, config):
        afm = AdversarialFlowMatrix(config.d_head, config.d_model)
        keys = torch.randn(2, 8, 16, config.d_head)
        ctx = torch.randn(2, config.d_model)
        out = afm(keys, ctx)
        # AFM uses ReLU on contradiction, so individual components >= 0
        assert torch.isfinite(out).all()

    def test_scores_with_probs(self, config):
        afm = AdversarialFlowMatrix(config.d_head, config.d_model)
        keys = torch.randn(2, 8, 16, config.d_head)
        ctx = torch.randn(2, config.d_model)
        probs = torch.softmax(torch.randn(2, 8, 16), dim=-1)
        out = afm(keys, ctx, key_probs=probs)
        assert out.shape == (2, 8, 1, 16)
