"""Tests for the Ramsey Partitioner."""

import torch
import pytest
from dstt import DSTTConfig
from dstt.partitioning import RamseyPartitioner
from dstt.utils import partition_count_to_heads, hardy_ramanujan_approx


class TestPartitionMath:
    """Tests for partition theory utilities."""

    def test_hardy_ramanujan_positive(self):
        for n in [10, 50, 100, 768]:
            assert hardy_ramanujan_approx(n) > 0

    def test_hardy_ramanujan_monotonic(self):
        vals = [hardy_ramanujan_approx(n) for n in range(1, 100)]
        for i in range(1, len(vals)):
            assert vals[i] >= vals[i - 1]

    def test_head_count_reasonable(self):
        h = partition_count_to_heads(768)
        assert 4 <= h <= 768 // 8

    def test_head_count_clamped(self):
        h = partition_count_to_heads(8, clamp_min=2)
        assert h >= 2


class TestRamseyPartitioner:
    """Tests for the RamseyPartitioner module."""

    @pytest.fixture
    def config(self):
        return DSTTConfig.tiny()

    def test_initial_assignments_cover_all_dims(self, config):
        rp = RamseyPartitioner(config)
        assignments = rp.assignments
        assert len(assignments) == config.d_model
        # Every dimension assigned to some head
        assert assignments.min() >= 0
        assert assignments.max() < rp.n_heads

    def test_get_head_dims_sum(self, config):
        rp = RamseyPartitioner(config)
        dims = rp.get_head_dims()
        assert sum(dims) == config.d_model
        assert len(dims) == rp.n_heads

    def test_get_head_indices(self, config):
        rp = RamseyPartitioner(config)
        indices = rp.get_head_indices()
        assert len(indices) == rp.n_heads
        all_idx = torch.cat(indices).sort()[0]
        expected = torch.arange(config.d_model)
        assert torch.equal(all_idx, expected)

    def test_recompute_preserves_coverage(self, config):
        rp = RamseyPartitioner(config)
        rp.recompute_partitions()
        dims = rp.get_head_dims()
        assert sum(dims) == config.d_model

    def test_forward_returns_indices(self, config):
        rp = RamseyPartitioner(config)
        indices = rp()
        assert isinstance(indices, list)
        assert len(indices) == rp.n_heads
