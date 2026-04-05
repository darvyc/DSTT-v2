"""Tests for RPMultiHeadAttention and DualFlowScoring."""

import torch
import pytest
from dstt import DSTTConfig, RPMultiHeadAttention
from dstt.flow_matrices import CorrectFlowMatrix, AdversarialFlowMatrix, DualFlowScoring


@pytest.fixture
def config():
    return DSTTConfig.tiny()


@pytest.fixture
def device():
    return torch.device("cpu")


class TestDualFlowScoring:
    """Tests for the CFM-AFM dual-flow attention scoring."""

    def test_cfm_output_shape(self, config):
        d_head = config.d_head
        cfm = CorrectFlowMatrix(d_head, config.d_model)
        keys = torch.randn(2, 8, 32, d_head)
        context = torch.randn(2, config.d_model)
        out = cfm(keys, context)
        assert out.shape == (2, 8, 1, 32)

    def test_afm_output_shape(self, config):
        d_head = config.d_head
        afm = AdversarialFlowMatrix(d_head, config.d_model)
        keys = torch.randn(2, 8, 32, d_head)
        context = torch.randn(2, config.d_model)
        out = afm(keys, context)
        assert out.shape == (2, 8, 1, 32)

    def test_dual_flow_output_shape(self, config):
        d_head = config.d_head
        dfs = DualFlowScoring(config, d_head)
        q = torch.randn(2, 8, 32, d_head)
        k = torch.randn(2, 8, 32, d_head)
        ctx = torch.randn(2, config.d_model)
        out = dfs(q, k, ctx)
        assert out.shape == (2, 8, 32, 32)

    def test_alpha_beta_are_learnable(self, config):
        d_head = config.d_head
        dfs = DualFlowScoring(config, d_head)
        assert dfs.alpha.requires_grad
        assert dfs.beta.requires_grad
        assert abs(dfs.alpha.item() - config.cfm_alpha_init) < 1e-6

    def test_dual_flow_with_mask(self, config):
        d_head = config.d_head
        dfs = DualFlowScoring(config, d_head)
        q = torch.randn(2, 8, 16, d_head)
        k = torch.randn(2, 8, 16, d_head)
        ctx = torch.randn(2, config.d_model)
        mask = torch.tril(torch.ones(1, 1, 16, 16))
        out = dfs(q, k, ctx, attention_mask=mask)
        # Upper triangle should be -inf
        assert torch.isinf(out[0, 0, 0, 1]) and out[0, 0, 0, 1] < 0

    def test_cfm_with_prev_state(self, config):
        d_head = config.d_head
        cfm = CorrectFlowMatrix(d_head, config.d_model)
        keys = torch.randn(2, 8, 32, d_head)
        context = torch.randn(2, config.d_model)
        prev = torch.randn(2, 32, d_head)
        out_with = cfm(keys, context, prev_state=prev)
        out_without = cfm(keys, context, prev_state=None)
        # Should differ when prev_state is provided
        assert not torch.allclose(out_with, out_without, atol=1e-5)


class TestRPMultiHeadAttention:
    """Tests for Ramsey-Partitioned Multi-Head Attention."""

    def test_output_shape(self, config):
        attn = RPMultiHeadAttention(config, layer_idx=0)
        x = torch.randn(2, 32, config.d_model)
        ctx = torch.randn(2, config.d_model)
        out = attn(x, ctx)
        assert out.shape == (2, 32, config.d_model)

    def test_causal_mask(self, config):
        attn = RPMultiHeadAttention(config, layer_idx=0)
        x = torch.randn(1, 8, config.d_model)
        ctx = torch.randn(1, config.d_model)
        mask = torch.tril(torch.ones(1, 1, 8, 8))
        out = attn(x, ctx, attention_mask=mask)
        assert out.shape == (1, 8, config.d_model)

    def test_gradient_flow(self, config):
        attn = RPMultiHeadAttention(config, layer_idx=0)
        x = torch.randn(2, 16, config.d_model, requires_grad=True)
        ctx = torch.randn(2, config.d_model)
        out = attn(x, ctx)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape
