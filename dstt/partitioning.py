"""
Ramsey Partitioner.

Partitions the model dimension into attention heads using
Ramsey-coherence clustering rather than equal-width slicing.

The partition ensures that dimensions assigned to the same head
are maximally coherent, while dimensions in different heads are
separated by the coherence threshold τ.
"""

from __future__ import annotations

import math
from typing import List, Tuple

import torch
import torch.nn as nn

from dstt.config import DSTTConfig
from dstt.utils import partition_count_to_heads


class RamseyPartitioner(nn.Module):
    """Partitions the model dimension into variable-width attention heads.

    Uses a coherence-threshold clustering criterion inspired by Ramsey
    theory: dimensions i and j are assigned to the same head iff their
    pairwise coherence exceeds threshold τ.

    The partitioner maintains a learnable coherence matrix and produces
    a deterministic partition assignment. The partition is recomputed
    only during EML updates (not during gradient steps).

    Attributes:
        d_model: Model dimension to partition.
        n_heads: Target number of heads (from Ramanujan-Hardy or config).
        threshold: Coherence threshold τ.
        assignments: Buffer storing dimension-to-head assignments.
    """

    def __init__(self, config: DSTTConfig):
        super().__init__()
        self.d_model = config.d_model
        self.threshold = config.coherence_threshold

        if config.use_ramsey_heads:
            self.n_heads = min(config.ramsey_head_count, config.n_heads)
        else:
            self.n_heads = config.n_heads

        # Learnable dimension embeddings for coherence computation
        self.dim_embeddings = nn.Parameter(
            torch.randn(config.d_model, 64) * 0.02
        )

        # Partition assignments (non-differentiable, updated by EML)
        self.register_buffer(
            "assignments",
            self._compute_initial_assignments()
        )

    def _compute_initial_assignments(self) -> torch.Tensor:
        """Compute initial equal-width partition as fallback.

        Returns:
            Tensor of shape (d_model,) with integer head assignments.
        """
        dims_per_head = self.d_model // self.n_heads
        remainder = self.d_model % self.n_heads
        assignments = []
        for h in range(self.n_heads):
            count = dims_per_head + (1 if h < remainder else 0)
            assignments.extend([h] * count)
        return torch.tensor(assignments, dtype=torch.long)

    @torch.no_grad()
    def recompute_partitions(self) -> None:
        """Recompute partition assignments from learned dim embeddings.

        Uses greedy coherence-threshold clustering:
        1. Compute pairwise cosine similarity of dimension embeddings.
        2. Build a union-find structure.
        3. Merge dimensions with similarity > τ.
        4. If too many clusters, merge smallest; if too few, split largest.

        This method is called by EML, not by gradient descent.
        """
        with torch.no_grad():
            emb = self.dim_embeddings.detach()
            emb_norm = emb / (emb.norm(dim=-1, keepdim=True) + 1e-8)
            sim = emb_norm @ emb_norm.T  # (d_model, d_model)

            # Union-find
            parent = list(range(self.d_model))

            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            def union(a, b):
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[ra] = rb

            # Merge dimensions with coherence > threshold
            for i in range(self.d_model):
                for j in range(i + 1, self.d_model):
                    if sim[i, j].item() > self.threshold:
                        union(i, j)

            # Collect clusters
            clusters: dict[int, list[int]] = {}
            for i in range(self.d_model):
                root = find(i)
                clusters.setdefault(root, []).append(i)

            cluster_list = list(clusters.values())

            # Adjust to target head count
            while len(cluster_list) > self.n_heads:
                # Merge two smallest clusters
                cluster_list.sort(key=len)
                merged = cluster_list[0] + cluster_list[1]
                cluster_list = [merged] + cluster_list[2:]

            while len(cluster_list) < self.n_heads:
                # Split largest cluster in half
                cluster_list.sort(key=len, reverse=True)
                largest = cluster_list[0]
                mid = len(largest) // 2
                cluster_list = [largest[:mid], largest[mid:]] + cluster_list[1:]

            # Build assignment tensor
            new_assignments = torch.zeros(self.d_model, dtype=torch.long)
            for head_idx, dims in enumerate(cluster_list):
                for d in dims:
                    new_assignments[d] = head_idx

            self.assignments.copy_(new_assignments)

    def get_head_dims(self) -> List[int]:
        """Return the width (number of dimensions) of each head.

        Returns:
            List of length n_heads with the dimension count per head.
        """
        counts = []
        for h in range(self.n_heads):
            counts.append(int((self.assignments == h).sum().item()))
        return counts

    def get_head_indices(self) -> List[torch.Tensor]:
        """Return the dimension indices assigned to each head.

        Returns:
            List of n_heads tensors, each containing the indices
            of dimensions assigned to that head.
        """
        indices = []
        for h in range(self.n_heads):
            idx = (self.assignments == h).nonzero(as_tuple=True)[0]
            indices.append(idx)
        return indices

    def forward(self) -> List[torch.Tensor]:
        """Return current head index assignments.

        Returns:
            List of index tensors, one per head.
        """
        return self.get_head_indices()
