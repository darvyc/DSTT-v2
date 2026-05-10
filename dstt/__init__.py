"""
DSTT-v2: Dynamic Semi-Trained Topology v2
====================================================

A lightweight sequence model architecture fusing combinatorial partitioning,
dual-flow attention, and evolutionary meta-optimisation into every
layer of the transformer stack.

Quick start::

    from dstt import DSTTConfig, DSTTv2

    config = DSTTConfig(d_model=256, n_layers=4, vocab_size=10000)
    model = DSTTv2(config)

    import torch
    x = torch.randint(0, 10000, (2, 128))
    logits = model(x)  # (2, 128, 10000)
"""

__version__ = "3.0.0"

from dstt.config import DSTTConfig
from dstt.model import DSTTv2, DSTTv2, DSTTBlock
from dstt.attention import LightweightTensorMixer, RPMultiHeadAttention
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
    "DSTTv2",
    "DSTTBlock",
    # Components
    "LightweightTensorMixer",
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
