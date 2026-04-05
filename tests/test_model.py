"""Tests for DSTTTransformer and DSTTBlock."""

import torch
import pytest
from dstt import DSTTConfig, DSTTTransformer, DSTTBlock
from dstt.losses import DSTTLoss
from dstt.utils import count_parameters


@pytest.fixture
def config():
    return DSTTConfig.tiny()


class TestDSTTBlock:

    def test_output_shape(self, config):
        block = DSTTBlock(config, layer_idx=0)
        x = torch.randn(2, 32, config.d_model)
        ctx = torch.randn(2, config.d_model)
        out = block(x, ctx)
        assert out.shape == (2, 32, config.d_model)

    def test_with_prev_state(self, config):
        block = DSTTBlock(config, layer_idx=0)
        x = torch.randn(2, 32, config.d_model)
        ctx = torch.randn(2, config.d_model)
        prev = torch.randn(2, 32, config.d_model)
        out = block(x, ctx, prev_state=prev)
        assert out.shape == (2, 32, config.d_model)

    def test_without_wittgenstein_gate(self):
        cfg = DSTTConfig.tiny()
        cfg.use_wittgenstein_gate = False
        block = DSTTBlock(cfg, layer_idx=0)
        x = torch.randn(2, 32, cfg.d_model)
        ctx = torch.randn(2, cfg.d_model)
        out = block(x, ctx)
        assert out.shape == (2, 32, cfg.d_model)


class TestDSTTTransformer:

    def test_output_shape(self, config):
        model = DSTTTransformer(config)
        x = torch.randint(0, config.vocab_size, (2, 64))
        logits = model(x)
        assert logits.shape == (2, 64, config.vocab_size)

    def test_single_token(self, config):
        model = DSTTTransformer(config)
        x = torch.randint(0, config.vocab_size, (1, 1))
        logits = model(x)
        assert logits.shape == (1, 1, config.vocab_size)

    def test_gradient_flow_full(self, config):
        model = DSTTTransformer(config)
        x = torch.randint(0, config.vocab_size, (2, 32))
        targets = torch.randint(0, config.vocab_size, (2, 32))
        logits = model(x)
        loss_fn = DSTTLoss(load_balance_weight=config.load_balance_weight)
        losses = loss_fn(logits, targets, model)
        losses["loss"].backward()
        # Check gradients exist on key parameters
        assert model.blocks[0].attention.q_proj.weight.grad is not None

    def test_param_count_positive(self, config):
        model = DSTTTransformer(config)
        assert model.get_num_params() > 0

    def test_repr(self, config):
        model = DSTTTransformer(config)
        r = repr(model)
        assert "DSTTTransformer" in r
        assert "dual_flow=True" in r

    def test_weight_tying(self, config):
        model = DSTTTransformer(config)
        assert model.lm_head.weight is model.embedding.token_embed.weight

    def test_causal_mask_generated(self, config):
        model = DSTTTransformer(config)
        # Should work without explicitly passing a mask
        x = torch.randint(0, config.vocab_size, (1, 16))
        logits = model(x)
        assert logits.shape == (1, 16, config.vocab_size)


class TestDSTTLoss:

    def test_loss_returns_dict(self, config):
        model = DSTTTransformer(config)
        x = torch.randint(0, config.vocab_size, (2, 32))
        targets = torch.randint(0, config.vocab_size, (2, 32))
        logits = model(x)
        loss_fn = DSTTLoss()
        result = loss_fn(logits, targets, model)
        assert "loss" in result
        assert "task_loss" in result
        assert "lb_loss" in result
        assert result["loss"].item() > 0
