"""
DSTT-T: Dynamic Semi-Trained Topology — Transformer
====================================================

A hybrid transformer architecture fusing combinatorial partitioning,
dual-flow attention, and evolutionary meta-optimisation into every
layer of the transformer stack.

Quick start::

    from dstt import DSTTConfig, DSTTTransformer

    config = DSTTConfig(d_model=256, n_layers=4, vocab_size=10000)
    model = DSTTTransformer(config)

    import torch
    x = torch.randint(0, 10000, (2, 128))
    logits = model(x)  # (2, 128, 10000)
"""

__version__ = "3.0.0"

from dstt.config import DSTTConfig
from dstt.model import DSTTTransformer, DSTTBlock
from dstt.attention import RPMultiHeadAttention
from dstt.embedding import FDMPEmbedding
from dstt.routing import ARMFeedForward
from dstt.gating import WittgensteinGate
from dstt.flow_matrices import CorrectFlowMatrix, AdversarialFlowMatrix, DualFlowScoring
from dstt.partitioning import RamseyPartitioner
from dstt.evolution import EvolutionaryMetaOptimiser, Chromosome
from dstt.losses import DSTTLoss, LoadBalanceLoss
from dstt.train_config import TrainConfig
from dstt.trainer import Trainer
from dstt.generate import generate, generate_text
from dstt.tokenizer import CharTokenizer, GPT2Tokenizer, get_tokenizer
from dstt.data import TextDataset, MemmapDataset, create_datasets

__all__ = [
    # Model
    "DSTTConfig",
    "DSTTTransformer",
    "DSTTBlock",
    # Components
    "RPMultiHeadAttention",
    "FDMPEmbedding",
    "ARMFeedForward",
    "WittgensteinGate",
    "CorrectFlowMatrix",
    "AdversarialFlowMatrix",
    "DualFlowScoring",
    "RamseyPartitioner",
    "EvolutionaryMetaOptimiser",
    "Chromosome",
    # Training
    "TrainConfig",
    "Trainer",
    "DSTTLoss",
    "LoadBalanceLoss",
    # Data
    "TextDataset",
    "MemmapDataset",
    "create_datasets",
    # Generation
    "generate",
    "generate_text",
    # Tokenizer
    "CharTokenizer",
    "GPT2Tokenizer",
    "get_tokenizer",
]
